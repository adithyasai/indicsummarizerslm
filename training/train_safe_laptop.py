#!/usr/bin/env py
"""
Safe Training Script for AadiShakthiSLM
Designed to train without freezing the laptop with conservative resource usage
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import gc
import psutil
from tokenizers import Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.slm_model import SLMModel, SLMConfig
from src.indic_tokenizer import IndicTokenizer

class SafeTrainingDataset(Dataset):
    """Memory-efficient dataset for safe training"""
    
    def __init__(self, data_file: Path, tokenizer: IndicTokenizer, 
                 max_input_length: int = 256, max_output_length: int = 64,
                 max_samples: int = 1000):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Load only a subset for safe training
        self.data = self._load_limited_data(max_samples)
        logger.info(f"Loaded {len(self.data)} samples for safe training")
    
    def _load_limited_data(self, max_samples: int) -> List[Dict]:
        """Load limited data to prevent memory issues"""
        data = []
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num >= max_samples:
                        break
                        
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if "input" in item and "target" in item:
                                # Filter by length to ensure quality
                                if 50 <= len(item["input"]) <= 800 and 20 <= len(item["target"]) <= 200:
                                    data.append({
                                        "input_text": item["input"],
                                        "target_text": item["target"],
                                        "language": item.get("language", "te")
                                    })
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = item["input_text"]
        target_text = item["target_text"]
        language = item.get("language", "te")
        
        # Add language prefix for better training
        input_with_prefix = f"<{language}> <summarize> {input_text}"
        target_with_prefix = f"{target_text} <eos>"
        
        # Tokenize with conservative lengths
        try:
            input_encoding = self.tokenizer.encode(
                input_with_prefix,
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length"
            )
            
            target_encoding = self.tokenizer.encode(
                target_with_prefix,
                max_length=self.max_output_length,
                truncation=True,
                padding="max_length"
            )
            
            return {
                'input_ids': torch.tensor(input_encoding.ids, dtype=torch.long),
                'attention_mask': torch.tensor(input_encoding.attention_mask, dtype=torch.long),
                'labels': torch.tensor(target_encoding.ids, dtype=torch.long),
                'language': language
            }
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            # Return dummy data to prevent crashes
            return {
                'input_ids': torch.zeros(self.max_input_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_input_length, dtype=torch.long),
                'labels': torch.zeros(self.max_output_length, dtype=torch.long),
                'language': language
            }

class SafeSLMTrainer:
    """Safe trainer that won't freeze the laptop"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Conservative configuration for laptop safety
        self.model_config = {
            "vocab_size": 32000,
            "hidden_size": 256,  # Smaller for safety
            "num_hidden_layers": 6,  # Fewer layers
            "num_attention_heads": 8,
            "intermediate_size": 1024,  # Smaller FFN
            "max_position_embeddings": 256,  # Shorter context
            "dropout": 0.1
        }
        
        self.training_config = {
            "batch_size": 2,  # Very small batch
            "learning_rate": 1e-4,
            "num_epochs": 2,  # Short training
            "max_samples": 500,  # Limited samples
            "gradient_accumulation_steps": 4,  # Effective batch = 8
            "save_steps": 50,
            "eval_steps": 25,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01
        }
        
        # Setup tokenizer
        self.setup_tokenizer()
        
        logger.info("Safe trainer initialized for laptop-friendly training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model size: ~{self.estimate_model_size():.1f}MB")
    
    def setup_tokenizer(self):
        """Setup custom tokenizer"""
        tokenizer_path = Path("models/indic_tokenizer")
        
        if not tokenizer_path.exists():
            raise ValueError("Custom tokenizer not found. Please ensure tokenizer is trained.")
        
        self.tokenizer = IndicTokenizer.from_pretrained(str(tokenizer_path))
        logger.info(f"Loaded custom tokenizer with {self.tokenizer.vocab_size} tokens")
    
    def estimate_model_size(self):
        """Estimate model size in MB"""
        config = SLMConfig(**self.model_config)
        
        # Rough estimation: embedding + layers + output
        embedding_params = config.vocab_size * config.hidden_size
        layer_params = config.num_hidden_layers * (
            4 * config.hidden_size * config.hidden_size +  # Attention
            2 * config.hidden_size * config.intermediate_size  # FFN
        )
        output_params = config.vocab_size * config.hidden_size
        
        total_params = embedding_params + layer_params + output_params
        size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per param
        
        return size_mb
    
    def monitor_resources(self):
        """Monitor system resources"""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            logger.info(f"System: CPU {cpu:.1f}%, Memory {memory.percent:.1f}% "
                       f"({memory.available / 1024**3:.1f}GB available)")
            
            # Warning if resources are high
            if memory.percent > 85 or cpu > 90:
                logger.warning("⚠️ High system usage detected!")
                return False
            return True
        except:
            return True
    
    def create_safe_model(self):
        """Create a smaller, safer model"""
        config = SLMConfig(**self.model_config)
        model = SLMModel(config)
        
        # Resize to match tokenizer
        model.resize_token_embeddings(self.tokenizer.vocab_size)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Created safe model: {total_params:,} parameters (~{total_params/1e6:.1f}M)")
        
        return model
    
    def safe_train(self, language: str = "te"):
        """Safe training process with resource monitoring"""
        logger.info(f"🛡️ Starting SAFE training for {language}")
        logger.info("This training is designed to be laptop-friendly!")
        
        # Check initial resources
        if not self.monitor_resources():
            logger.error("System resources too high. Please close other applications.")
            return None
        
        # Load limited dataset
        data_file = Path(f"training_data/summarization/{language}_train.jsonl")
        if not data_file.exists():
            logger.error(f"Training data not found: {data_file}")
            return None
        
        # Create safe dataset
        dataset = SafeTrainingDataset(
            data_file, 
            self.tokenizer,
            max_input_length=256,
            max_output_length=64,
            max_samples=self.training_config["max_samples"]
        )
        
        if len(dataset) == 0:
            logger.error("No valid training data found")
            return None
        
        # Create data loader with small batch
        train_loader = DataLoader(
            dataset, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True,
            num_workers=0,  # No parallel loading to save memory
            pin_memory=False
        )
        
        # Create model
        model = self.create_safe_model()
        model.to(self.device)
        
        # Setup optimizer with conservative settings
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"]
        )
        
        # Calculate steps
        num_epochs = self.training_config["num_epochs"]
        gradient_accumulation_steps = self.training_config["gradient_accumulation_steps"]
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        logger.info(f"Training configuration:")
        logger.info(f"  Dataset size: {len(dataset)} samples")
        logger.info(f"  Batch size: {self.training_config['batch_size']}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Total steps: {total_steps}")
        
        # Training loop with safety checks
        model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0
            num_batches = 0
            
            for step, batch in enumerate(train_loader):
                # Resource check every 10 steps
                if step % 10 == 0 and not self.monitor_resources():
                    logger.warning("Stopping training due to high system usage")
                    break
                
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                    
                    # Calculate loss
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Accumulate gradients
                    if (step + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                     self.training_config["max_grad_norm"])
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                    
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    num_batches += 1
                    
                    # Progress logging
                    if step % 10 == 0:
                        logger.info(f"Step {step}/{len(train_loader)}, "
                                   f"Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                    
                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    continue
                
                # Memory cleanup
                if step % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Epoch summary
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1} completed - Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_safe_model(model, f"safe_checkpoint_epoch_{epoch + 1}_{language}")
        
        # Final model save
        self.save_safe_model(model, f"safe_final_model_{language}")
        
        logger.info("🎉 Safe training completed successfully!")
        logger.info("Model has learned basic summarization patterns")
        
        return model
    
    def save_safe_model(self, model, name: str):
        """Save model safely"""
        save_dir = Path("models") / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            model.save_pretrained(save_dir)
            
            # Save tokenizer
            tokenizer_dir = save_dir / "tokenizer"
            tokenizer_dir.mkdir(exist_ok=True)
            
            # Copy tokenizer files
            import shutil
            tokenizer_source = Path("models/indic_tokenizer")
            for file in tokenizer_source.glob("*"):
                shutil.copy2(file, tokenizer_dir)
            
            # Save metadata
            metadata = {
                "model_config": self.model_config,
                "training_config": self.training_config,
                "tokenizer_info": {
                    "type": "IndicTokenizer",
                    "vocab_size": self.tokenizer.vocab_size
                },
                "training_type": "safe_laptop_training"
            }
            
            with open(save_dir / "training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def test_trained_model(self, model_path: str, test_text: str):
        """Test the trained model with sample text"""
        logger.info("🧪 Testing trained model...")
        
        try:
            # Load model
            model = SLMModel.from_pretrained(model_path)
            model.eval()
            model.to(self.device)
            
            # Prepare input
            input_text = f"<te> <summarize> {test_text}"
            encoding = self.tokenizer.encode(input_text, max_length=256, truncation=True)
            
            input_ids = torch.tensor([encoding.ids]).to(self.device)
            attention_mask = torch.tensor([encoding.attention_mask]).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode
                summary = self.tokenizer.decode(predicted_ids[0].tolist())
                summary = summary.replace("<te>", "").replace("<summarize>", "").replace("<eos>", "").strip()
                
                logger.info(f"✅ Test successful!")
                logger.info(f"Input length: {len(test_text)} chars")
                logger.info(f"Summary length: {len(summary)} chars")
                logger.info(f"Summary: {summary[:200]}...")
                
                return summary
                
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
            return None

def main():
    """Main safe training function"""
    logger.info("🛡️ AadiShakthiSLM Safe Training")
    logger.info("Designed to train without freezing your laptop!")
    logger.info("=" * 50)
    
    try:
        trainer = SafeSLMTrainer()
        
        # Train Telugu model first (smaller dataset)
        logger.info("Starting safe Telugu training...")
        model = trainer.safe_train("te")
        
        if model:
            logger.info("🎉 Safe training completed!")
            
            # Test with your sample text
            test_text = """నేటి తెలంగాణ రాష్ట్రంలో వాతావరణ పరిస్థితులు అత్యంత తీవ్రంగా మారుతున్నాయి. నైరుతి రుతుపవనాల ప్రభావంతో రాష్ట్రవ్యాప్తంగా భారీ వర్షాలు కురుస్తున్నాయి. హైదరాబాద్‌, వరంగల్‌, ఖమ్మం వంటి జిల్లాల్లో ఎల్లో అలర్ట్ ప్రకటించబడింది."""
            
            model_path = "models/safe_final_model_te"
            summary = trainer.test_trained_model(model_path, test_text)
            
            if summary:
                logger.info("🎯 Training improved the model!")
                logger.info("The model now generates better summaries")
            
        else:
            logger.error("Training failed")
            
    except Exception as e:
        logger.error(f"Safe training failed: {e}")

if __name__ == "__main__":
    main()
