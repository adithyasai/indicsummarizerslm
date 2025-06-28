#!/usr/bin/env python3
"""
Clean Incremental Training for AadiShakti SLM
Simplified version without Unicode issues
"""

import os
import json
import torch
import gc
import psutil
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel, GPT2Config, GPT2Tokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for incremental training"""
    # Resource limits
    max_memory_usage: float = 0.7
    max_cpu_usage: float = 0.8
    batch_size: int = 2
    max_length: int = 256
    
    # Training parameters
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_steps_per_chunk: int = 200  # Smaller for safety
    save_every_n_steps: int = 50
    
    # Data processing
    chunk_size: int = 500
    max_examples_per_session: int = 2000
    
    # Directories
    model_dir: str = "models/incremental_slm"
    checkpoint_dir: str = "models/checkpoints"
    data_dir: str = "data/processed"

class SimpleDataset(Dataset):
    """Simple dataset for training"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class SimpleTrainer:
    """Simple trainer without complex monitoring"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.current_step = 0
        
        # Create directories
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info("Simple Trainer initialized")
    
    def load_tokenizer(self):
        """Load tokenizer"""
        tokenizer_path = "models/indic_tokenizer"
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded: {len(self.tokenizer)} tokens")
        return self.tokenizer
    
    def initialize_model(self):
        """Initialize model"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "latest")
        
        if os.path.exists(checkpoint_path):
            logger.info("Loading existing model...")
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
            
            # Load training state
            state_file = os.path.join(checkpoint_path, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.current_step = state.get('step', 0)
        else:
            logger.info("Creating new model...")
            config = GPT2Config(
                vocab_size=len(self.tokenizer),
                n_positions=self.config.max_length,
                n_embd=256,
                n_layer=6,
                n_head=8,
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )
            self.model = GPT2LMHeadModel(config)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        logger.info(f"Model ready on {device}")
        return self.model
    
    def load_training_data(self, chunk_idx: int) -> List[str]:
        """Load training data"""
        data_files = [
            "data/processed/telugu_texts.json",
            "data/processed/hindi_texts.json",
            "data/processed/mixed_texts.json"
        ]
        
        all_texts = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, dict) and 'texts' in data:
                        all_texts.extend(data['texts'])
                    elif isinstance(data, list):
                        all_texts.extend(data)
                        
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        # Create sample data if none exists
        if not all_texts:
            all_texts = [
                "తెలుగు భాష చాలా అందమైన భాష.",
                "हिंदी भारत की राजभाषा है।",
                "भारत एक विविधताओं से भरा देश है।",
                "తెలంగాణ రాష్ట్రం భారతదేశంలో ఒక ముఖ్యమైన రాష్ట్రం.",
                "प्रौद्योगिकी के क्षेत्र में भारत तेजी से आगे बढ़ रहा है।"
            ] * 100  # Repeat to have enough data
        
        # Get chunk
        start_idx = chunk_idx * self.config.chunk_size
        end_idx = start_idx + self.config.chunk_size
        chunk_texts = all_texts[start_idx:end_idx]
        
        logger.info(f"Loaded chunk {chunk_idx}: {len(chunk_texts)} texts")
        return chunk_texts
    
    def train_chunk(self, texts: List[str]) -> bool:
        """Train on a chunk of data"""
        if not texts:
            return False
        
        logger.info(f"Training on {len(texts)} texts...")
        
        # Create dataset
        dataset = SimpleDataset(texts, self.tokenizer, self.config.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=self.config.batch_size,
            save_steps=self.config.save_every_n_steps,
            save_total_limit=2,
            logging_steps=25,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps_per_chunk,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )
        
        try:
            # Train
            trainer.train()
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Cleanup
            del trainer, dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Chunk training completed")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def save_checkpoint(self):
        """Save checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "latest")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save state
        state = {
            'step': self.current_step,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved at step {self.current_step}")
    
    def run_training(self, max_chunks: int = 5):
        """Run complete training"""
        logger.info("Starting training...")
        logger.info(f"Max chunks: {max_chunks}")
        
        # Initialize
        self.load_tokenizer()
        self.initialize_model()
        
        successful_chunks = 0
        
        for chunk_idx in range(max_chunks):
            logger.info(f"=== CHUNK {chunk_idx + 1}/{max_chunks} ===")
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                logger.warning("High memory usage, pausing...")
                time.sleep(30)
                gc.collect()
            
            # Load and train
            texts = self.load_training_data(chunk_idx)
            
            if not texts:
                logger.info("No more data")
                break
            
            if self.train_chunk(texts):
                successful_chunks += 1
                self.current_step += len(texts) // self.config.batch_size
            
            # Pause between chunks
            if chunk_idx < max_chunks - 1:
                logger.info("Pausing before next chunk...")
                time.sleep(30)
        
        # Final save
        self.save_checkpoint()
        
        # Copy to main model directory
        try:
            import shutil
            shutil.copytree(
                os.path.join(self.config.checkpoint_dir, "latest"),
                self.config.model_dir,
                dirs_exist_ok=True
            )
            logger.info(f"Model copied to {self.config.model_dir}")
        except Exception as e:
            logger.warning(f"Could not copy to main directory: {e}")
        
        logger.info("Training completed!")
        logger.info(f"Successful chunks: {successful_chunks}/{max_chunks}")
        logger.info(f"Model saved to: {self.config.model_dir}")

def main():
    """Main function"""
    logger.info("AadiShakti SLM - Clean Training")
    logger.info("=" * 50)
    
    # Create trainer
    config = TrainingConfig()
    trainer = SimpleTrainer(config)
    
    # Run training
    try:
        trainer.run_training(max_chunks=5)  # Start with just 5 chunks
        logger.info("SUCCESS: Training completed")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
