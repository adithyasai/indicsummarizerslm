#!/usr/bin/env python3
"""
Fast Summarization Training - Focus training on summarization task
Using existing processed data to train for summarization specifically
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import json
import logging
from pathlib import Path
import os
from datasets import Dataset
from typing import List, Dict
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationTrainer:
    """Fast trainer focused on summarization task"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load existing tokenizer
        tokenizer_path = "models/indic_tokenizer"
        if os.path.exists(tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
            logger.info(f"✅ Loaded tokenizer from {tokenizer_path}")
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            
        # Load existing model or create new one
        model_path = "models/checkpoints/latest"
        if os.path.exists(model_path):
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            logger.info(f"✅ Loaded existing model from {model_path}")
        else:
            # Create new model with existing tokenizer vocab size
            from transformers import GPT2Config
            config = GPT2Config(
                vocab_size=len(self.tokenizer),
                n_positions=512,
                n_embd=768,
                n_layer=12,
                n_head=12
            )
            self.model = GPT2LMHeadModel(config)
            logger.info("✅ Created new model")
            
        self.model.to(self.device)
        
        # Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_and_prepare_data(self) -> List[Dict]:
        """Load Telugu and Hindi data and prepare for summarization training"""
        training_examples = []
        
        # Load Telugu data
        telugu_file = "data/processed/telugu_texts_simple.json"
        if os.path.exists(telugu_file):
            with open(telugu_file, 'r', encoding='utf-8') as f:
                telugu_data = json.load(f)
                logger.info(f"Loaded {len(telugu_data)} Telugu texts")
                
                # Create summarization examples from Telugu data
                for text in telugu_data[:500]:  # Use first 500 for speed
                    if len(text) > 200:  # Only texts long enough to summarize
                        training_examples.extend(self.create_summarization_examples(text))
        
        # Load Hindi data
        hindi_file = "data/processed/hindi_texts_simple.json"
        if os.path.exists(hindi_file):
            with open(hindi_file, 'r', encoding='utf-8') as f:
                hindi_data = json.load(f)
                logger.info(f"Loaded {len(hindi_data)} Hindi texts")
                
                # Create summarization examples from Hindi data
                for text in hindi_data[:500]:  # Use first 500 for speed
                    if len(text) > 200:
                        training_examples.extend(self.create_summarization_examples(text))
        
        logger.info(f"Created {len(training_examples)} summarization training examples")
        return training_examples
    
    def create_summarization_examples(self, text: str) -> List[Dict]:
        """Create summarization training examples from a text"""
        examples = []
        
        # Split text into sentences
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 3:
            return examples
        
        # Create multiple summarization examples
        for i in range(min(3, len(sentences) - 2)):  # Create up to 3 examples per text
            # Take a portion of the text as input
            start_idx = i
            end_idx = min(start_idx + random.randint(3, min(6, len(sentences) - start_idx)), len(sentences))
            
            input_sentences = sentences[start_idx:end_idx]
            input_text = '. '.join(input_sentences) + '.'
            
            # Create a summary (first sentence + key information)
            if len(input_sentences) >= 2:
                # Simple extractive summary: first sentence + one key sentence
                summary_sentences = [input_sentences[0]]
                if len(input_sentences) > 2:
                    # Add the longest sentence as it likely contains key info
                    longest_sentence = max(input_sentences[1:], key=len)
                    if longest_sentence not in summary_sentences:
                        summary_sentences.append(longest_sentence)
                
                summary = '. '.join(summary_sentences) + '.'
                
                # Create training example
                prompt = f"సారాంశం చేయండి: {input_text}\n\nసారాంశం: {summary}"
                
                examples.append({
                    'input_text': input_text,
                    'summary': summary,
                    'prompt': prompt
                })
        
        return examples
    
    def prepare_dataset(self, examples: List[Dict]) -> Dataset:
        """Prepare dataset for training"""
        
        def tokenize_function(example):
            # Tokenize the full prompt (input + summary)
            tokenized = self.tokenizer(
                example['prompt'],
                truncation=True,
                padding=True,
                max_length=400,
                return_tensors="pt"
            )
            
            # Labels are the same as input_ids for language modeling
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        # Convert to dataset
        dataset = Dataset.from_list(examples)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            batched=False
        )
        
        return tokenized_dataset
    
    def train_fast(self, num_epochs: int = 2):
        """Fast training focused on summarization"""
        logger.info("🚀 Starting fast summarization training...")
        
        # Load data
        examples = self.load_and_prepare_data()
        if not examples:
            logger.error("No training examples found!")
            return False
        
        # Prepare dataset
        dataset = self.prepare_dataset(examples)
        
        # Training arguments - optimized for speed
        training_args = TrainingArguments(
            output_dir="models/summarization_checkpoints",
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,  # Small batch size to avoid memory issues
            gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
            warmup_steps=50,
            logging_steps=25,
            save_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=False,  # Disable fp16 to avoid issues
            learning_rate=5e-5,
            weight_decay=0.01,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to=[],  # Disable wandb/tensorboard for speed
        )
        
        # Data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train
        try:
            logger.info(f"Training on {len(dataset)} examples...")
            trainer.train()
            
            # Save the trained model
            output_dir = "models/summarization_model"
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"✅ Training complete! Model saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

def quick_train():
    """Quick training function"""
    trainer = SummarizationTrainer()
    success = trainer.train_fast(num_epochs=1)  # Just 1 epoch for speed
    return success

if __name__ == "__main__":
    print("🎯 Fast Summarization Training")
    print("=" * 50)
    
    success = quick_train()
    
    if success:
        print("✅ Training successful!")
        print("Model saved to: models/summarization_model")
        print("You can now use this model for better summarization")
    else:
        print("❌ Training failed")
        print("Check the logs for details")
