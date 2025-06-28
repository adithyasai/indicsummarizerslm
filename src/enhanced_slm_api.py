#!/usr/bin/env py
"""
Enhanced SLM Model Wrapper with Custom IndicTokenizer Integration
Provides improved summarization using our trained tokenizer and model architecture
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, List
import unicodedata
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.slm_model import SLMModel, SLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSLMSummarizer:
    """
    Enhanced SLM Summarizer with custom IndicTokenizer integration
    Provides improved quality over the basic extractive approach
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # Try to load custom tokenizer
        self.load_custom_tokenizer()
        
        # Try to load trained model if available
        if model_path and Path(model_path).exists():
            self.load_trained_model(model_path)
        else:
            logger.info("No trained model found. Using enhanced extractive summarization with custom tokenizer.")
    
    def load_custom_tokenizer(self):
        """Load our custom IndicTokenizer"""
        try:
            from src.indic_tokenizer import IndicTokenizer
            tokenizer_path = Path("models/indic_tokenizer")
            
            if tokenizer_path.exists():
                self.tokenizer = IndicTokenizer.from_pretrained(str(tokenizer_path))
                logger.info(f"✅ Loaded custom IndicTokenizer with {self.tokenizer.vocab_size} tokens")
                return True
            else:
                logger.warning("Custom tokenizer not found, falling back to basic tokenization")
                return False
                
        except Exception as e:
            logger.error(f"Error loading custom tokenizer: {e}")
            return False
    
    def load_trained_model(self, model_path: str):
        """Load a trained SLM model"""
        try:
            # Load model configuration
            config_file = Path(model_path) / "config.json"
            if config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                config = SLMConfig(**config_dict)
            else:
                # Default configuration
                config = SLMConfig(
                    vocab_size=self.tokenizer.vocab_size if self.tokenizer else 32000,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12
                )
            
            # Load model
            self.model = SLMModel(config)
            
            # Load weights if available
            model_file = Path(model_path) / "pytorch_model.bin"
            if model_file.exists():
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"✅ Loaded trained model from {model_path}")
            else:
                logger.info("Model weights not found, using initialized model")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            self.model = None
    
    def detect_language(self, text: str) -> str:
        """Enhanced language detection using Unicode script analysis"""
        if not text:
            return "unknown"
        
        # Count characters by script
        telugu_chars = sum(1 for char in text if '\u0c00' <= char <= '\u0c7f')
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097f')
        english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return "unknown"
        
        # Calculate percentages
        telugu_pct = telugu_chars / total_chars
        hindi_pct = hindi_chars / total_chars
        english_pct = english_chars / total_chars
        
        # Language detection logic
        if telugu_pct > 0.3:
            return "te"
        elif hindi_pct > 0.3:
            return "hi"
        elif english_pct > 0.7:
            return "en"
        else:
            return "mixed"
    
    def enhanced_extractive_summarize(self, text: str, max_length: int = 150, language: str = "auto") -> Dict:
        """Enhanced extractive summarization with custom tokenizer"""
        
        if language == "auto":
            detected_language = self.detect_language(text)
        else:
            detected_language = language
        
        # Split into sentences with better handling for Indic languages
        sentences = self.split_sentences(text, detected_language)
        
        if len(sentences) <= 2:
            return {
                "summary": text[:max_length] + ("..." if len(text) > max_length else ""),
                "confidence": 0.5,
                "method": "truncation",
                "language": detected_language
            }
        
        # Enhanced sentence scoring
        sentence_scores = self.score_sentences(sentences, detected_language)
        
        # Select top sentences
        top_sentences = self.select_top_sentences(sentences, sentence_scores, max_length)
        
        # Reorder sentences to maintain original order
        summary_sentences = sorted(top_sentences, key=lambda x: sentences.index(x))
        summary = " ".join(summary_sentences)
        
        # Calculate confidence based on score distribution
        if sentence_scores:
            score_std = torch.std(torch.tensor(list(sentence_scores.values()))).item()
            confidence = min(0.9, 0.5 + score_std * 0.5)  # Higher std = better discrimination
        else:
            confidence = 0.5
        
        return {
            "summary": summary,
            "confidence": confidence,
            "method": "enhanced_extractive",
            "language": detected_language,
            "sentence_count": len(summary_sentences),
            "compression_ratio": len(summary) / len(text) if text else 0
        }
    
    def split_sentences(self, text: str, language: str) -> List[str]:
        """Improved sentence splitting for Indic languages"""
        # Basic sentence splitting with support for Indic punctuation
        sentences = re.split(r'[।.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def score_sentences(self, sentences: List[str], language: str) -> Dict[str, float]:
        """Enhanced sentence scoring with custom tokenizer"""
        if not sentences:
            return {}
        
        sentence_scores = {}
        
        # Calculate word frequencies using custom tokenizer if available
        word_freq = {}
        all_text = " ".join(sentences)
        
        if self.tokenizer:
            # Use custom tokenizer for better word segmentation
            try:
                encoding = self.tokenizer.encode(all_text)
                tokens = [self.tokenizer.decode([token_id]) for token_id in encoding.ids]
                
                for token in tokens:
                    token = token.strip()
                    if len(token) > 1 and not token.startswith('<'):  # Skip special tokens
                        word_freq[token] = word_freq.get(token, 0) + 1
                        
            except Exception as e:
                logger.warning(f"Error using custom tokenizer for scoring: {e}")
                # Fallback to simple word splitting
                words = all_text.lower().split()
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
        else:
            # Fallback: simple word splitting
            words = all_text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score each sentence
        for sentence in sentences:
            score = 0
            word_count = 0
            
            if self.tokenizer:
                try:
                    encoding = self.tokenizer.encode(sentence)
                    tokens = [self.tokenizer.decode([token_id]) for token_id in encoding.ids]
                    
                    for token in tokens:
                        token = token.strip()
                        if len(token) > 1 and not token.startswith('<'):
                            score += word_freq.get(token, 0)
                            word_count += 1
                except:
                    # Fallback scoring
                    words = sentence.lower().split()
                    for word in words:
                        score += word_freq.get(word, 0)
                        word_count += 1
            else:
                words = sentence.lower().split()
                for word in words:
                    score += word_freq.get(word, 0)
                    word_count += 1
            
            # Normalize by sentence length
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
            else:
                sentence_scores[sentence] = 0
        
        return sentence_scores
    
    def select_top_sentences(self, sentences: List[str], scores: Dict[str, float], max_length: int) -> List[str]:
        """Select top sentences based on scores and length constraints"""
        # Sort sentences by score
        sorted_sentences = sorted(sentences, key=lambda x: scores.get(x, 0), reverse=True)
        
        selected = []
        total_length = 0
        
        for sentence in sorted_sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed max_length
            if total_length + sentence_length + len(selected) * 2 <= max_length:  # +2 for ". "
                selected.append(sentence)
                total_length += sentence_length
            
            # Stop if we have enough content
            if total_length >= max_length * 0.8:  # 80% of max length
                break
        
        # Ensure at least one sentence
        if not selected and sentences:
            selected = [sentences[0]]
        
        return selected
    
    def abstractive_summarize(self, text: str, max_length: int = 150, language: str = "auto") -> Dict:
        """Abstractive summarization using trained model (if available)"""
        if not self.model or not self.tokenizer:
            logger.info("Trained model not available, falling back to extractive summarization")
            return self.enhanced_extractive_summarize(text, max_length, language)
        
        try:
            if language == "auto":
                detected_language = self.detect_language(text)
            else:
                detected_language = language
            
            # Prepare input with language prefix
            input_text = f"<{detected_language}> <summarize> {text}"
            
            # Tokenize
            encoding = self.tokenizer.encode(
                input_text,
                max_length=1024,
                truncation=True,
                padding=False
            )
            
            input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Simple greedy decoding (can be improved with beam search)
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode output
                summary = self.tokenizer.decode(predicted_ids[0].tolist())
                
                # Clean up summary
                summary = summary.replace(f"<{detected_language}>", "").replace("<summarize>", "")
                summary = summary.replace("<eos>", "").replace("<pad>", "").strip()
                
                return {
                    "summary": summary[:max_length],
                    "confidence": 0.8,
                    "method": "abstractive",
                    "language": detected_language
                }
                
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {e}")
            return self.enhanced_extractive_summarize(text, max_length, language)
    
    def summarize(self, text: str, max_length: int = 150, method: str = "auto", language: str = "auto") -> Dict:
        """Main summarization method with improved quality"""
        
        if not text or len(text.strip()) < 20:
            return {
                "summary": text,
                "confidence": 0.1,
                "method": "passthrough",
                "language": "unknown"
            }
        
        # Choose method
        if method == "auto":
            # Use abstractive if model is available, otherwise extractive
            if self.model and self.tokenizer:
                method = "abstractive"
            else:
                method = "extractive"
        
        if method == "abstractive":
            return self.abstractive_summarize(text, max_length, language)
        else:
            return self.enhanced_extractive_summarize(text, max_length, language)

# Global instance for API
enhanced_summarizer = None

def get_summarizer():
    """Get or create the enhanced summarizer instance"""
    global enhanced_summarizer
    if enhanced_summarizer is None:
        # Try to load a trained model
        model_paths = [
            "models/final_model_summarization_te",
            "models/final_model_summarization_hi", 
            "models/best_model_summarization_te",
            "models/best_model_summarization_hi"
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        enhanced_summarizer = EnhancedSLMSummarizer(model_path)
    
    return enhanced_summarizer
