#!/usr/bin/env py
"""
Enhanced Training Script for AadiShakthiSLM
Implements training loop for Telugu and Hindi text summarization using collected data
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import time
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.slm_model import SLMModel, SLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummarizationDataset(Dataset):
    """Enhanced dataset class for summarization training"""
    
    def __init__(self, data_file: Path, tokenizer: PreTrainedTokenizer, max_input_length: int = 512, max_output_length: int = 128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Load data
        self.examples = []
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "input" in data and "target" in data:
                                self.examples.append({
                                    "input": data["input"],
                                    "target": data["target"],
                                    "language": data.get("language", "unknown")
                                })
                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_file}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input text
        inputs = self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target summary
        targets = self.tokenizer(
            example["target"],
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'language': example["language"]
        }

class LanguageModelingDataset(Dataset):
    """Dataset for language modeling (pre-training)"""
    
    def __init__(self, data_file: Path, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load text data
        self.texts = []
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) > 50:  # Minimum length
                        self.texts.append(line)
        
        logger.info(f"Loaded {len(self.texts)} texts from {data_file}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For language modeling, labels are the same as input_ids
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class AadiShakthiTrainer:
    """Main trainer class for AadiShakthiSLM"""
    
    def __init__(self, config_path: str = "training_data/training_config.json"):
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_device()
        self.setup_tokenizer()
        
    def load_config(self):
        """Load training configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.model_config = config.get("model_config", {})
                self.training_config = config.get("training_config", {})
                self.is_sample_data = config.get("sample_data", False)
        else:
            # Default configuration
            self.model_config = {
                "vocab_size": 50000,
                "hidden_size": 512,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 2048,
                "max_position_embeddings": 512,
                "dropout": 0.1
            }
            self.training_config = {
                "batch_size": 8,
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "warmup_steps": 1000,
                "save_steps": 5000,
                "eval_steps": 1000
            }
            self.is_sample_data = False
        
        logger.info(f"Loaded configuration: Sample data = {self.is_sample_data}")
    
    def setup_device(self):
        """Setup training device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def setup_tokenizer(self):
        """Setup tokenizer for multilingual support"""
        try:
            # Use multilingual BERT tokenizer as base
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            logger.info("Loaded multilingual BERT tokenizer")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def load_datasets(self, data_dir: str = "training_data"):
        """Load training datasets"""
        data_path = Path(data_dir)
        
        # Check available datasets
        sum_dir = data_path / "summarization"
        lm_dir = data_path / "language_modeling"
        
        datasets = {}
        
        # Load summarization datasets
        if sum_dir.exists():
            for lang in ["te", "hi"]:
                train_file = sum_dir / f"{lang}_train.jsonl"
                val_file = sum_dir / f"{lang}_val.jsonl"
                
                if train_file.exists():
                    train_dataset = SummarizationDataset(
                        train_file, 
                        self.tokenizer,
                        max_input_length=self.model_config.get("max_position_embeddings", 512),
                        max_output_length=128
                    )
                    datasets[f"summarization_{lang}_train"] = train_dataset
                
                if val_file.exists():
                    val_dataset = SummarizationDataset(
                        val_file,
                        self.tokenizer,
                        max_input_length=self.model_config.get("max_position_embeddings", 512),
                        max_output_length=128
                    )
                    datasets[f"summarization_{lang}_val"] = val_dataset
        
        # Load language modeling datasets
        if lm_dir.exists():
            for lang in ["te", "hi"]:
                train_file = lm_dir / f"{lang}_train.txt"
                val_file = lm_dir / f"{lang}_val.txt"
                
                if train_file.exists():
                    train_dataset = LanguageModelingDataset(
                        train_file,
                        self.tokenizer,
                        max_length=self.model_config.get("max_position_embeddings", 512)
                    )
                    datasets[f"language_modeling_{lang}_train"] = train_dataset
                
                if val_file.exists():
                    val_dataset = LanguageModelingDataset(
                        val_file,
                        self.tokenizer,
                        max_length=self.model_config.get("max_position_embeddings", 512)
                    )
                    datasets[f"language_modeling_{lang}_val"] = val_dataset
        
        logger.info(f"Loaded {len(datasets)} datasets")
        for name, dataset in datasets.items():
            logger.info(f"  {name}: {len(dataset)} examples")
        
        return datasets
    
    def create_model(self):
        """Create SLM model"""
        config = SLMConfig(**self.model_config)
        model = SLMModel(config)
        
        # Update vocab size based on tokenizer
        if hasattr(self.tokenizer, 'vocab_size'):
            config.vocab_size = self.tokenizer.vocab_size
            model.resize_token_embeddings(self.tokenizer.vocab_size)
        
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def train_step(self, model, batch, optimizer, scheduler, criterion):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Reshape for loss calculation
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        return loss.item()
    
    def evaluate(self, model, dataloader, criterion):
        """Evaluate model on validation data"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
                num_batches += 1
        
        model.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self, task: str = "summarization", language: str = "te", epochs: int = None):
        """Main training loop"""
        logger.info(f"Starting training for {task} in {language}")
        
        # Load datasets
        datasets = self.load_datasets()
        
        # Find appropriate datasets
        train_key = f"{task}_{language}_train"
        val_key = f"{task}_{language}_val"
        
        if train_key not in datasets:
            logger.error(f"Training dataset not found: {train_key}")
            logger.info(f"Available datasets: {list(datasets.keys())}")
            return
        
        train_dataset = datasets[train_key]
        val_dataset = datasets.get(val_key, None)
        
        # Create data loaders
        batch_size = self.training_config.get("batch_size", 8)
        if self.is_sample_data:
            batch_size = min(batch_size, 2)  # Smaller batch for sample data
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        
        # Create model
        model = self.create_model()
        model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.training_config.get("learning_rate", 2e-4),
            weight_decay=self.training_config.get("weight_decay", 0.01)
        )
        
        num_epochs = epochs or self.training_config.get("num_epochs", 3)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = self.training_config.get("warmup_steps", total_steps // 10)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Training loop
        logger.info(f"Training for {num_epochs} epochs with {len(train_loader)} steps per epoch")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            model.train()
            total_loss = 0
            num_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(model, batch, optimizer, scheduler, criterion)
                total_loss += loss
                num_steps += 1
                
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Log periodically
                if step % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss:.4f}")
            
            avg_train_loss = total_loss / num_steps
            logger.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self.evaluate(model, val_loader, criterion)
                logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                self.save_model(model, f"checkpoint_epoch_{epoch + 1}")
        
        logger.info("Training completed!")
        
        # Save final model
        self.save_model(model, "final_model")
        
        return model
    
    def save_model(self, model, name: str):
        """Save model and tokenizer"""
        save_dir = Path("models") / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(save_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save config
        config_file = save_dir / "training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_config": self.model_config,
                "training_config": self.training_config
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model saved to {save_dir}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train AadiShakthiSLM")
    parser.add_argument("--task", choices=["summarization", "language_modeling"], 
                       default="summarization", help="Training task")
    parser.add_argument("--language", choices=["te", "hi"], default="te",
                       help="Language to train on")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--config", default="training_data/training_config.json",
                       help="Training configuration file")
    parser.add_argument("--data-dir", default="training_data",
                       help="Training data directory")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AadiShakthiTrainer(args.config)
    
    # Start training
    model = trainer.train(
        task=args.task,
        language=args.language,
        epochs=args.epochs
    )
    
    if model:
        logger.info("🎉 Training completed successfully!")
        logger.info("Your enhanced model is ready to test!")
    else:
        logger.error("❌ Training failed!")

if __name__ == "__main__":
    main()
