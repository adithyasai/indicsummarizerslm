#!/usr/bin/env py
"""
Production SLM Training Script for AadiShakthiSLM
Integrates the custom IndicTokenizer with comprehensive Telugu and Hindi training
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import get_linear_schedule_with_warmup
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import time
import argparse
from tokenizers import Tokenizer

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.slm_model import SLMModel, SLMConfig
from src.indic_tokenizer import IndicTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionDataset(Dataset):
    """Production dataset class optimized for large-scale training"""
    
    def __init__(self, data_file: Path, tokenizer: IndicTokenizer, 
                 max_input_length: int = 512, max_output_length: int = 128, 
                 task_type: str = "summarization"):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.task_type = task_type
        
        # Load and cache data
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} samples from {data_file.name}")
    
    def _load_data(self) -> List[Dict]:
        """Load and parse JSONL data efficiently"""
        data = []
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            
                            # Validate required fields based on task type
                            if self.task_type == "summarization":
                                if "input" in item and "target" in item:
                                    data.append({
                                        "input_text": item["input"],
                                        "target_text": item["target"],
                                        "language": item.get("language", "unknown")
                                    })
                            elif self.task_type == "language_modeling":
                                if "text" in item:
                                    data.append({
                                        "text": item["text"],
                                        "language": item.get("language", "unknown")
                                    })
                                    
                        except json.JSONDecodeError as e:
                            if line_num % 1000 == 0:  # Log occasionally
                                logger.warning(f"Invalid JSON at line {line_num}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error loading data from {self.data_file}: {e}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.task_type == "summarization":
            return self._process_summarization_item(item)
        else:
            return self._process_lm_item(item)
    
    def _process_summarization_item(self, item):
        """Process summarization training item"""
        input_text = item["input_text"]
        target_text = item["target_text"]
        language = item.get("language", "te")
        
        # Add language and task prefixes for better training
        input_with_prefix = f"<{language}> <summarize> {input_text}"
        target_with_prefix = f"{target_text} <eos>"
        full_text = input_with_prefix + " " + target_with_prefix
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        full_ids = full_ids[:self.max_input_length]
        attention_mask = [1] * len(full_ids)
        if len(full_ids) < self.max_input_length:
            pad_length = self.max_input_length - len(full_ids)
            full_ids += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
        # Create labels: mask input part, only predict target
        input_ids = self.tokenizer.encode(input_with_prefix, add_special_tokens=True)
        input_len = len(input_ids)
        labels = [-100] * input_len + full_ids[input_len:]
        labels = labels[:self.max_input_length]
        if len(labels) < self.max_input_length:
            labels += [-100] * (self.max_input_length - len(labels))
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    def _process_lm_item(self, item):
        """Process language modeling item"""
        text = item["text"]
        language = item.get("language", "te")
        
        # Add language prefix
        text_with_prefix = f"<{language}> {text}"
        
        # Tokenize
        encoding = self.tokenizer.encode(
            text_with_prefix,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length"
        )
        
        return {
            'input_ids': torch.tensor(encoding.ids, dtype=torch.long),
            'attention_mask': torch.tensor(encoding.attention_mask, dtype=torch.long),
            'labels': torch.tensor(encoding.ids, dtype=torch.long),  # For LM, labels = input_ids
            'language': language
        }

class JsonlIterableDataset(IterableDataset):
    """Iterable dataset that streams JSONL records for large files"""
    def __init__(self, file_path, tokenizer, max_input_length, max_output_length=None, task_type="summarization"):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.task_type = task_type

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item.get('text') or item.get('document') or item.get('content')
                if self.task_type == 'summarization':
                    inputs = self.tokenizer.encode(text, truncation=True, max_length=self.max_input_length, return_tensors='pt')
                    targets = self.tokenizer.encode(item.get('summary', ''), truncation=True, max_length=self.max_output_length, return_tensors='pt')
                    yield {'input_ids': inputs.squeeze(), 'labels': targets.squeeze()}
                else:
                    inputs = self.tokenizer.encode(line, truncation=True, max_length=self.max_input_length, return_tensors='pt')
                    yield {'input_ids': inputs.squeeze(), 'labels': inputs.squeeze()}

class ProductionSLMTrainer:
    """Production-ready trainer for AadiShakthiSLM with custom tokenizer integration"""
    
    def __init__(self, config_path: str = "training_data/training_config.json"):
        self.config_path = Path(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.load_config()
        
        # Setup tokenizer (our custom IndicTokenizer)
        self.setup_custom_tokenizer()
        
        # Update model config with tokenizer vocab size
        self.model_config["vocab_size"] = self.tokenizer.get_vocab_size()
        
        logger.info(f"Production trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Tokenizer vocab size: {self.tokenizer.get_vocab_size()}")
    
    def load_config(self):
        """Load training configuration optimized for production"""
        default_config = {
            "model_config": {
                "vocab_size": 32000,  # Will be updated with actual tokenizer size
                "hidden_size": 768,   # Larger for better performance
                "num_hidden_layers": 12,  # Deeper model
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 1024,  # Longer context
                "dropout": 0.1,
                "layer_norm_eps": 1e-12
            },
            "training_config": {
                "batch_size": 16,  # Larger batches for stability
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "warmup_steps": 1000,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "save_steps": 2000,
                "eval_steps": 500,
                "logging_steps": 100,
                "gradient_accumulation_steps": 4  # Effective batch size = 64
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Update with production settings
                self.model_config = {**default_config["model_config"], **config.get("model_config", {})}
                self.training_config = {**default_config["training_config"], **config.get("training_config", {})}
        else:
            self.model_config = default_config["model_config"]
            self.training_config = default_config["training_config"]
        
        logger.info("Loaded production training configuration")
    
    def setup_custom_tokenizer(self):
        """Setup our custom IndicTokenizer"""
        tokenizer_path = Path("models/indic_tokenizer")
        
        if not tokenizer_path.exists():
            raise ValueError(f"Custom tokenizer not found at {tokenizer_path}. Please run tokenizer training first.")
        
        # Load the custom IndicTokenizer
        self.tokenizer = IndicTokenizer.load(str(tokenizer_path))
        logger.info(f"Loaded custom IndicTokenizer with {self.tokenizer.get_vocab_size()} tokens")
        self.model_config["vocab_size"] = self.tokenizer.get_vocab_size()
        logger.info(f"Tokenizer vocab size: {self.tokenizer.get_vocab_size()}")
    
    def create_model(self):
        """Create SLM model with production configuration"""
        config = SLMConfig(**self.model_config)
        model = SLMModel(config)
        
        # Initialize embeddings properly for our custom tokenizer
        if hasattr(self.tokenizer, 'get_vocab_size'):
            model.resize_token_embeddings(self.tokenizer.get_vocab_size())
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Created production SLM model:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        return model
    
    def load_datasets(self, data_dir: str = "training_data", task: str = "summarization", language: str = "te"):
        """Load production datasets (patched to use XL-Sum for Telugu summarization)"""
        data_path = Path(data_dir)
        datasets = {}

        if task == "summarization" and language == "te":
            # Always use the new XL-Sum data for Telugu
            train_file = data_path / "summarization" / "xlsum_te_train.jsonl"
            val_file = data_path / "summarization" / "te_val.jsonl"
        else:
            sum_dir = data_path / "summarization"
            train_file = sum_dir / f"{language}_train.jsonl"
            val_file = sum_dir / f"{language}_val.jsonl"

        if train_file.exists():
            train_dataset = ProductionDataset(
                train_file,
                self.tokenizer,
                max_input_length=self.model_config["max_position_embeddings"],
                max_output_length=128,
                task_type="summarization"
            )
            datasets["train"] = train_dataset

        if val_file.exists():
            val_dataset = ProductionDataset(
                val_file,
                self.tokenizer,
                max_input_length=self.model_config["max_position_embeddings"],
                max_output_length=128,
                task_type="summarization"
            )
            datasets["val"] = val_dataset

        return datasets
    
    def _convert_text_to_jsonl(self, text_file: Path, language: str) -> Path:
        """Convert text file to JSONL for consistent processing"""
        jsonl_file = text_file.with_suffix('.jsonl')
        
        if not jsonl_file.exists():
            logger.info(f"Converting {text_file.name} to JSONL format...")
            
            with open(text_file, 'r', encoding='utf-8') as f_in, \
                 open(jsonl_file, 'w', encoding='utf-8') as f_out:
                
                current_text = ""
                for line in f_in:
                    line = line.strip()
                    if line:
                        current_text += line + " "
                    else:
                        # End of document
                        if current_text.strip():
                            item = {
                                "text": current_text.strip(),
                                "language": language
                            }
                            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                            current_text = ""
                
                # Handle last document
                if current_text.strip():
                    item = {
                        "text": current_text.strip(),
                        "language": language
                    }
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return jsonl_file
    
    def train(self, task: str = "summarization", language: str = "te", epochs: int = None):
        """Production training loop with comprehensive monitoring"""
        logger.info(f"🚀 Starting production training: {task} for {language}")
        
        # Load datasets
        datasets = self.load_datasets(task=task, language=language)
        
        if "train" not in datasets:
            logger.error(f"No training data found for {task} in {language}")
            return None
        
        train_dataset = datasets["train"]
        val_dataset = datasets.get("val", None)
        
        # Create data loaders with optimized settings
        batch_size = self.training_config["batch_size"]
        gradient_accumulation_steps = self.training_config.get("gradient_accumulation_steps", 1)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,  # Parallel data loading
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        # Create model
        model = self.create_model()
        model.to(self.device)
        
        # Setup optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate training steps
        num_epochs = epochs or self.training_config["num_epochs"]
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = self.training_config["warmup_steps"]
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Training state
        global_step = 0
        best_val_loss = float('inf')
        training_stats = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": []
        }
        
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Ensure attention_mask is [batch_size, 1, 1, seq_len]
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(1).unsqueeze(1)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # DEBUG: Print unique values in labels to check for out-of-bounds
                if global_step == 0:
                    print('DEBUG unique label values:', torch.unique(labels))
                
                # Create loss function here for debugging
                criterion = nn.CrossEntropyLoss(ignore_index=-100)
                print('DEBUG criterion ignore_index:', criterion.ignore_index)
                
                # Calculate loss
                # Workaround: replace -100 with 0, compute loss, then mask out those positions
                labels_for_loss = labels.clone()
                mask = (labels_for_loss == -100)
                labels_for_loss[mask] = 0  # 0 is a valid token id, won't affect masked positions
                logits_t = logits.transpose(1, 2)
                raw_loss = criterion(logits_t, labels_for_loss)
                # Zero out loss for masked positions
                if raw_loss.dim() > 0:
                    raw_loss = raw_loss * (~mask)
                    loss = raw_loss.sum() / (~mask).sum().clamp(min=1)
                else:
                    loss = raw_loss
                loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradients
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.training_config["max_grad_norm"]
                    )
                    
                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # Statistics
                epoch_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Periodic logging
                if global_step % self.training_config["logging_steps"] == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    training_stats["learning_rates"].append(current_lr)
                    
                    logger.info(f"Step {global_step}: Loss = {loss.item() * gradient_accumulation_steps:.4f}, LR = {current_lr:.2e}")
            
            # Calculate average training loss
            avg_train_loss = epoch_loss / num_batches
            training_stats["train_losses"].append(avg_train_loss)
            
            logger.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_loader:
                val_loss = self.evaluate(model, val_loader, criterion)
                training_stats["val_losses"].append(val_loss)
                
                logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(model, f"best_model_{task}_{language}")
                    logger.info(f"New best model saved! Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                self.save_model(model, f"checkpoint_epoch_{epoch + 1}_{task}_{language}")
                logger.info(f"Checkpoint saved for epoch {epoch + 1}")
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Training completed!")
        logger.info(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        # Save final model
        self.save_model(model, f"final_model_{task}_{language}")
        
        # Save training statistics
        stats_file = Path("models") / f"training_stats_{task}_{language}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, indent=2)
        
        logger.info(f"Training statistics saved to {stats_file}")
        
        return model
    
    def evaluate(self, model, dataloader, criterion):
        """Comprehensive model evaluation"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Workaround: replace -100 with 0, compute loss, then mask out those positions
                labels_for_loss = labels.clone()
                mask = (labels_for_loss == -100)
                labels_for_loss[mask] = 0
                logits_t = logits.transpose(1, 2)
                raw_loss = criterion(logits_t, labels_for_loss)
                if raw_loss.dim() > 0:
                    raw_loss = raw_loss * (~mask)
                    loss = raw_loss.sum() / (~mask).sum().clamp(min=1)
                else:
                    loss = raw_loss
                
                total_loss += loss.item()
                num_batches += 1
        
        model.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_model(self, model, name: str):
        """Save model with comprehensive metadata"""
        save_dir = Path("models") / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(save_dir, safe_serialization=False)
        
        # Save tokenizer (copy our custom tokenizer files)
        tokenizer_dir = save_dir / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        
        # Copy tokenizer files
        import shutil
        tokenizer_source = Path("models/indic_tokenizer")
        if tokenizer_source.exists():
            for file in tokenizer_source.glob("*"):
                shutil.copy2(file, tokenizer_dir)
        
        # Save training metadata
        metadata = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "tokenizer_info": {
                "type": "IndicTokenizer",
                "vocab_size": self.tokenizer.get_vocab_size(),
                "path": str(tokenizer_dir)
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(self.device)
        }
        
        metadata_file = save_dir / "training_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model and metadata saved to {save_dir}")

def main():
    """Main entry point for production training"""
    parser = argparse.ArgumentParser(description="Production SLM Training for AadiShakthiSLM")
    parser.add_argument("--task", choices=["summarization", "language_modeling"], 
                       default="summarization", help="Training task")
    parser.add_argument("--language", choices=["te", "hi"], default="te",
                       help="Language to train on")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--config", default="training_data/training_config.json",
                       help="Training configuration file")
    
    args = parser.parse_args()
    
    logger.info("🚀 AadiShakthiSLM Production Training")
    logger.info("=====================================")
    logger.info(f"Task: {args.task}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Config: {args.config}")
    
    try:
        # Initialize trainer
        trainer = ProductionSLMTrainer(args.config)
        
        # Start training
        model = trainer.train(
            task=args.task,
            language=args.language,
            epochs=args.epochs
        )
        
        if model:
            logger.info("🎉 Production training completed successfully!")
            logger.info("Your enhanced AadiShakthiSLM is ready!")
            
            # Print next steps
            logger.info("\n📋 Next Steps:")
            logger.info("1. Test the trained model with sample texts")
            logger.info("2. Evaluate on test datasets")
            logger.info("3. Deploy for inference")
            logger.info("4. Consider training on the other language")
        else:
            logger.error("❌ Training failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
