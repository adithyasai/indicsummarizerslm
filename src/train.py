"""
Training script for AadiShakthiSLM
This is a template for training the model on Telugu and Hindi datasets
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationDataset(Dataset):
    """Dataset class for summarization training"""
    
    def __init__(self, texts: List[str], summaries: List[str], tokenizer, max_input_length: int = 512, max_output_length: int = 128):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target summary
        targets = self.tokenizer(
            summary,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def load_xlsum_data(language: str = "telugu", split: str = "train", max_samples: Optional[int] = None):
    """Load XL-Sum dataset for specified language"""
    try:
        logger.info(f"Loading XL-Sum dataset for {language}...")
        
        # Language code mapping
        lang_codes = {
            "telugu": "telugu",
            "hindi": "hindi",
            "english": "english"
        }
        
        if language not in lang_codes:
            raise ValueError(f"Unsupported language: {language}")
        
        # Load dataset
        dataset = load_dataset("csebuetnlp/xlsum", lang_codes[language], split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        texts = dataset['text']
        summaries = dataset['summary']
        
        logger.info(f"Loaded {len(texts)} samples for {language}")
        return texts, summaries
        
    except Exception as e:
        logger.error(f"Error loading XL-Sum data: {e}")
        return [], []

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the SLM model"""
    
    model.to(device)
    model.train()
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log every 100 steps
            if batch_idx % 100 == 0:
                logger.info(f"Step {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        # Validation
        if val_dataloader:
            val_loss = evaluate_model(model, val_dataloader, device)
            logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

def evaluate_model(model, dataloader, device):
    """Evaluate the model on validation data"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(dataloader)

def main():
    """Main training function"""
    
    # Load configuration
    with open('config/model_config.json', 'r') as f:
        config = json.load(f)
    
    logger.info("Starting AadiShakthiSLM training...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # TODO: Implement actual training when datasets are available
    logger.info("Training script template ready!")
    logger.info("To start training:")
    logger.info("1. Ensure datasets are downloaded")
    logger.info("2. Configure tokenizer and model")
    logger.info("3. Uncomment and modify the training code below")
    
    # Example training code (commented out for now)
    """
    # Load data
    train_texts, train_summaries = load_xlsum_data("telugu", "train", max_samples=1000)
    val_texts, val_summaries = load_xlsum_data("telugu", "validation", max_samples=100)
    
    # Create datasets
    from models.slm_model import SLMModel, SLMConfig
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    train_dataset = SummarizationDataset(train_texts, train_summaries, tokenizer)
    val_dataset = SummarizationDataset(val_texts, val_summaries, tokenizer)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    config = SLMConfig()
    model = SLMModel(config)
    
    # Train model
    train_model(model, train_dataloader, val_dataloader, num_epochs=3)
    
    # Save model
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    """

if __name__ == "__main__":
    main()
