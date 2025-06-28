#!/usr/bin/env py
"""
Incremental Training System for AadiShakti SLM
Processes data in manageable chunks to avoid system freezes
"""

import os
import json
import torch
import gc
import psutil
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('incremental_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Fix encoding issues on Windows
import sys
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except:
        pass  # Ignore if already wrapped
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for incremental training"""
    # Resource limits
    max_memory_usage: float = 0.7  # 70% of available RAM
    max_cpu_usage: float = 0.8     # 80% CPU usage
    batch_size: int = 2            # Small batch size for stability
    max_length: int = 256          # Reduced sequence length
    
    # Training parameters
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_steps_per_chunk: int = 500  # Small chunks
    save_every_n_steps: int = 100
    
    # Data processing
    chunk_size: int = 1000         # Process 1000 examples at a time
    max_examples_per_session: int = 5000  # Limit per training session
    
    # Directories
    model_dir: str = "models/incremental_slm"
    checkpoint_dir: str = "models/checkpoints"
    data_dir: str = "data/processed"

class TeluguHindiDataset(Dataset):
    """Dataset for Telugu and Hindi text processing"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize with proper truncation
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

class ResourceMonitor:
    """Monitor system resources during training"""
    
    def __init__(self, max_memory: float = 0.7, max_cpu: float = 0.8):
        self.max_memory = max_memory
        self.max_cpu = max_cpu
        self.process = psutil.Process()
        
    def check_resources(self) -> Dict[str, float]:
        """Check current resource usage"""
        # Memory usage
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1) / 100.0
        
        # Process memory
        process_memory = self.process.memory_percent() / 100.0
        
        return {
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent,
            'process_memory': process_memory
        }
    
    def should_pause(self) -> bool:
        """Check if training should pause due to resource constraints"""
        resources = self.check_resources()
        
        if resources['memory_percent'] > self.max_memory:
            logger.warning(f"High memory usage: {resources['memory_percent']:.2%}")
            return True
            
        if resources['cpu_percent'] > self.max_cpu:
            logger.warning(f"High CPU usage: {resources['cpu_percent']:.2%}")
            return True
            
        return False
    
    def wait_for_resources(self, max_wait: int = 300):
        """Wait for resources to free up"""
        logger.info("Waiting for system resources to free up...")
        start_time = time.time()
        
        while self.should_pause() and (time.time() - start_time) < max_wait:
            time.sleep(10)
            gc.collect()  # Force garbage collection
            
        if time.time() - start_time >= max_wait:
            logger.warning("Resource wait timeout - proceeding with caution")

class IncrementalTrainer:
    """Incremental training system"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.monitor = ResourceMonitor(config.max_memory_usage, config.max_cpu_usage)
        
        # Create directories
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.data_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.current_step = 0        
        logger.info("Incremental Trainer initialized")
        logger.info(f"   Max memory usage: {config.max_memory_usage:.1%}")
        logger.info(f"   Max CPU usage: {config.max_cpu_usage:.1%}")
        logger.info(f"   Batch size: {config.batch_size}")
        logger.info(f"   Chunk size: {config.chunk_size}")
    
    def load_tokenizer(self):
        """Load the custom Indic tokenizer"""
        tokenizer_path = "models/indic_tokenizer"
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
        # Ensure special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded: {len(self.tokenizer)} tokens")
        return self.tokenizer
      def initialize_model(self):
        """Initialize or load the model"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "latest")
        
        if os.path.exists(checkpoint_path):
            logger.info("Loading existing model checkpoint...")
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
            
            # Load training state
            state_file = os.path.join(checkpoint_path, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.current_step = state.get('step', 0)
        else:
            logger.info("Initializing new model...")
            config = GPT2Config(
                vocab_size=len(self.tokenizer),
                n_positions=self.config.max_length,
                n_embd=256,      # Smaller embedding dimension
                n_layer=6,       # Fewer layers for stability
                n_head=8,        # Fewer attention heads
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )
            self.model = GPT2LMHeadModel(config)
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        logger.info(f"Model initialized on {device}")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        return self.model
    
    def load_data_chunk(self, chunk_idx: int) -> List[str]:
        """Load a chunk of training data"""
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
                        if isinstance(data, list):
                            all_texts.extend(data)
                        elif isinstance(data, dict):
                            all_texts.extend(data.get('texts', []))
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        if not all_texts:
            # Fallback: create sample data from raw files
            all_texts = self.extract_sample_data()
        
        # Get chunk
        start_idx = chunk_idx * self.config.chunk_size
        end_idx = start_idx + self.config.chunk_size
        chunk_texts = all_texts[start_idx:end_idx]
        
        logger.info(f"📥 Loaded chunk {chunk_idx}: {len(chunk_texts)} texts")
        return chunk_texts
    
    def extract_sample_data(self) -> List[str]:
        """Extract sample data from raw datasets"""
        logger.info("📚 Extracting sample data from raw datasets...")
        
        sample_texts = []
        
        # Try to extract from various sources
        raw_data_paths = [
            "data/raw/telugu_wikipedia",
            "data/raw/hindi_wikipedia", 
            "data/raw/cc100_telugu",
            "data/raw/cc100_hindi"
        ]
        
        for data_path in raw_data_paths:
            if os.path.exists(data_path):
                # Look for text files
                for file_path in Path(data_path).rglob("*.txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Split into sentences and take good ones
                            sentences = content.split('.')
                            good_sentences = [
                                s.strip() for s in sentences 
                                if len(s.strip()) > 50 and len(s.strip()) < 500
                            ]
                            sample_texts.extend(good_sentences[:100])  # Limit per file
                            
                            if len(sample_texts) >= 2000:  # Enough for initial training
                                break
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")
                        continue
                
                if len(sample_texts) >= 2000:
                    break
        
        # If still no data, create minimal training data
        if not sample_texts:
            sample_texts = [
                "తెలుగు భాష చాలా అందమైన భాష.",
                "హిंदी भाषा भारत की राजभाषा है.",
                "भारत एक विविधताओं से भरा देश है.",
                "తెలంగాణ రాష్ట్రం భారతదేశంలో ఒక ముఖ్యమైన రాష్ట్రం.",
                "प्रौद्योगिकी के क्षेत्र में भारत तेजी से आगे बढ़ रहा है."
            ] * 400  # Repeat to have enough data
        
        logger.info(f"📝 Extracted {len(sample_texts)} sample texts")
        return sample_texts
    
    def train_chunk(self, texts: List[str]) -> Dict:
        """Train on a single chunk of data"""
        if not texts:
            return {"status": "no_data", "steps": 0}
        
        logger.info(f"🎯 Training on chunk with {len(texts)} texts")
        
        # Create dataset
        dataset = TeluguHindiDataset(texts, self.tokenizer, self.config.max_length)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=self.config.batch_size,
            save_steps=self.config.save_every_n_steps,
            save_total_limit=3,
            logging_steps=50,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps_per_chunk,
            dataloader_pin_memory=False,  # Reduce memory usage
            dataloader_num_workers=0,     # Avoid multiprocessing issues
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
        
        # Train with resource monitoring
        steps_completed = 0
        try:
            # Check resources before training
            if self.monitor.should_pause():
                self.monitor.wait_for_resources()
            
            logger.info("🏃 Starting chunk training...")
            train_result = trainer.train()
            steps_completed = train_result.training_loss
            
            # Save after each chunk
            self.save_checkpoint()
            
            # Clean up
            del trainer, dataset, dataloader
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"✅ Chunk training completed: {steps_completed} steps")
            
        except Exception as e:
            logger.error(f"❌ Chunk training failed: {e}")
            return {"status": "error", "error": str(e), "steps": steps_completed}
        
        return {"status": "success", "steps": steps_completed}
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "latest")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        state = {
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved at step {self.current_step}")
      def run_incremental_training(self, max_chunks: int = 10, pause_between_chunks: int = 30):
        """Run incremental training across multiple chunks"""
        logger.info("Starting incremental training...")
        logger.info(f"   Max chunks: {max_chunks}")
        logger.info(f"   Pause between chunks: {pause_between_chunks}s")
        
        # Initialize
        self.load_tokenizer()
        self.initialize_model()
        
        total_steps = 0
        successful_chunks = 0
        
        for chunk_idx in range(max_chunks):
            logger.info(f"\n📊 === CHUNK {chunk_idx + 1}/{max_chunks} ===")
            
            # Check resources before starting chunk
            resources = self.monitor.check_resources()
            logger.info(f"💻 Resources - Memory: {resources['memory_percent']:.1%}, CPU: {resources['cpu_percent']:.1%}")
            
            if self.monitor.should_pause():
                logger.info("⏸️ Pausing for resources...")
                self.monitor.wait_for_resources()
            
            # Load and train chunk
            try:
                texts = self.load_data_chunk(chunk_idx)
                
                if not texts:
                    logger.info(f"📭 No more data at chunk {chunk_idx}")
                    break
                
                result = self.train_chunk(texts)
                
                if result["status"] == "success":
                    successful_chunks += 1
                    total_steps += result["steps"]
                    self.current_step += result["steps"]
                    
                    logger.info(f"✅ Chunk {chunk_idx + 1} completed successfully")
                else:
                    logger.error(f"❌ Chunk {chunk_idx + 1} failed: {result.get('error', 'Unknown error')}")
                
                # Pause between chunks to let system recover
                if chunk_idx < max_chunks - 1:
                    logger.info(f"⏱️ Pausing {pause_between_chunks}s before next chunk...")
                    time.sleep(pause_between_chunks)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in chunk {chunk_idx}: {e}")
                continue
        
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
            logger.info(f"📋 Model copied to {self.config.model_dir}")
        except Exception as e:
            logger.warning(f"Could not copy to main directory: {e}")
        
        logger.info("\n🎉 Incremental training completed!")
        logger.info(f"   Total chunks processed: {successful_chunks}/{max_chunks}")
        logger.info(f"   Total training steps: {total_steps}")
        logger.info(f"   Final model saved to: {self.config.model_dir}")

def main():
    """Main training function"""
    logger.info("AadiShakti SLM - Incremental Training")
    logger.info("=" * 60)
    
    # Create training configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = IncrementalTrainer(config)
    
    # Run training
    try:
        trainer.run_incremental_training(
            max_chunks=20,  # Process up to 20 chunks
            pause_between_chunks=60  # 1 minute pause between chunks
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
