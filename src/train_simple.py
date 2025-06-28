#!/usr/bin/env py
"""
Simple Working Training Script for AadiShakthiSLM
A simplified version that works with our sample data to improve summarization
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSummarizationModel(nn.Module):
    """Simple seq2seq model for summarization"""
    
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Use pre-trained multilingual BERT as encoder
        self.encoder = AutoModel.from_pretrained("bert-base-multilingual-cased")
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(tokenizer))
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Encode
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded = encoder_outputs.last_hidden_state
        
        # Simple pooling (take mean of all tokens)
        pooled = encoded.mean(dim=1)  # (batch_size, hidden_size)
        
        # Expand for sequence generation (simplified)
        seq_length = labels.size(1) if labels is not None else 64
        expanded = pooled.unsqueeze(1).expand(-1, seq_length, -1)
        
        # Decode
        logits = self.decoder(expanded)
        
        if labels is not None:
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}

class SummarizationDataset(Dataset):
    """Simple dataset for summarization"""
    
    def __init__(self, data_file: Path, tokenizer, max_input_length: int = 256, max_output_length: int = 64):
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
        
        # Tokenize input
        inputs = self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
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
            'labels': targets['input_ids'].squeeze()
        }

def train_simple_model(language: str = "te", epochs: int = 5):
    """Train a simple summarization model"""
    logger.info(f"Training simple summarization model for {language}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Load data
    train_file = Path(f"training_data/summarization/{language}_train.jsonl")
    val_file = Path(f"training_data/summarization/{language}_val.jsonl")
    
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        return None
    
    train_dataset = SummarizationDataset(train_file, tokenizer)
    val_dataset = SummarizationDataset(val_file, tokenizer) if val_file.exists() else None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Small batch for sample data
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset else None
    
    # Create model
    model = SimpleSummarizationModel(tokenizer)
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 1 == 0:  # Log every step for small dataset
                logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask, labels)
                    val_loss += outputs['loss'].item()
            
            val_loss /= len(val_loader)
            logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            model.train()
    
    # Save model
    save_dir = Path("models/simple_summarizer")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_dir / f"{language}_model.pt")
    tokenizer.save_pretrained(save_dir)
    
    # Save configuration
    config = {
        "language": language,
        "model_type": "simple_summarizer",
        "max_input_length": 256,
        "max_output_length": 64
    }
    
    with open(save_dir / f"{language}_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Model saved to {save_dir}")
    return model

def test_model(model, tokenizer, text: str):
    """Test the trained model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        logits = outputs['logits']
        
        # Generate summary (simple greedy decoding)
        predicted_ids = torch.argmax(logits, dim=-1)
        summary = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    
    return summary

def main():
    """Main entry point"""
    logger.info("🚀 Starting Simple SLM Training")
    
    # Train Telugu model
    logger.info("Training Telugu summarization model...")
    te_model = train_simple_model("te", epochs=10)
    
    if te_model:
        # Test with sample text
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        test_text = "హైదరాబాద్‌లో నేడు అంతర్జాతీయ సమావేశం జరుగుతోంది. ఈ సమావేశంలో ప్రపంచవ్యాప్తంగా వివిధ దేశాల ప్రతినిధులు పాల్గొంటున్నారు."
        
        summary = test_model(te_model, tokenizer, test_text)
        logger.info(f"Test summary: {summary}")
    
    # Train Hindi model
    logger.info("Training Hindi summarization model...")
    hi_model = train_simple_model("hi", epochs=10)
    
    logger.info("✅ Training completed!")
    logger.info("Now you can test the improved summarization!")

if __name__ == "__main__":
    main()
