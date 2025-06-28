#!/usr/bin/env py
"""
Comprehensive Tokenizer Training Script for AadiShakthiSLM
Trains the custom IndicTokenizer on all available Telugu and Hindi datasets
"""

import os
import json
import logging
import gzip
import lzma
from pathlib import Path
from typing import List, Iterator
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from indic_tokenizer import IndicTokenizer, TokenizerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tokenizer_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """Utility class to load various dataset formats"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load_jsonl(self, file_path: Path, text_fields: List[str] = None) -> Iterator[str]:
        """Load text from JSONL files"""
        if text_fields is None:
            text_fields = ['text', 'content', 'article', 'summary', 'target']
            
        logger.info(f"Loading JSONL: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0 and line_num > 0:
                        logger.info(f"Processed {line_num} lines from {file_path.name}")
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract text from various fields
                        for field in text_fields:
                            if field in data and data[field]:
                                text = str(data[field]).strip()
                                if text and len(text) > 10:  # Filter out very short texts
                                    yield text
                                    
                        # Handle nested structures
                        if 'translation' in data:
                            for lang_code, text in data['translation'].items():
                                if text and len(str(text)) > 10:
                                    yield str(text)
                                    
                    except (json.JSONDecodeError, KeyError) as e:
                        if line_num < 10:  # Only log first few errors
                            logger.warning(f"Error parsing line {line_num} in {file_path.name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def load_compressed_text(self, file_path: Path) -> Iterator[str]:
        """Load text from compressed files (.xz, .gz)"""
        logger.info(f"Loading compressed file: {file_path}")
        
        try:
            if file_path.suffix == '.xz':
                opener = lzma.open
            elif file_path.suffix == '.gz':
                opener = gzip.open
            else:
                # Try regular text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if line_num % 10000 == 0 and line_num > 0:
                            logger.info(f"Processed {line_num} lines from {file_path.name}")
                        
                        text = line.strip()
                        if text and len(text) > 10:
                            yield text
                return
            
            with opener(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0 and line_num > 0:
                        logger.info(f"Processed {line_num} lines from {file_path.name}")
                    
                    text = line.strip()
                    if text and len(text) > 10:
                        yield text
                        
        except Exception as e:
            logger.error(f"Error loading compressed file {file_path}: {e}")
    
    def load_wiki_dump(self, wiki_dir: Path) -> Iterator[str]:
        """Load text from Wikipedia dump directories"""
        logger.info(f"Loading Wikipedia dump: {wiki_dir}")
        
        for file_path in wiki_dir.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and len(content) > 50:
                        # Split into paragraphs
                        paragraphs = content.split('\n\n')
                        for para in paragraphs:
                            para = para.strip()
                            if para and len(para) > 20:
                                yield para
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
    
    def get_all_training_texts(self, max_texts_per_source: int = 100000) -> List[str]:
        """Load all available training texts from the dataset directory"""
        all_texts = []
        
        # Define dataset sources and their loaders
        sources = [
            # JSONL files
            ("samanantar_telugu_50k.jsonl", "jsonl", ["te"]),
            ("samanantar_hindi_50k.jsonl", "jsonl", ["hi"]),
            ("wikilingua_hindi_full.jsonl", "jsonl", ["text", "summary"]),
            ("xlsum_telugu_combined.jsonl", "jsonl", ["text", "summary"]),
            ("xlsum_hindi_combined.jsonl", "jsonl", ["text", "summary"]),
            
            # Compressed text files
            ("te.txt.xz", "compressed", None),
            ("hi.txt.xz", "compressed", None),
            
            # Wikipedia dumps
            ("wiki_te", "wiki", None),
            ("wiki_hi", "wiki", None),
        ]
        
        for source_name, source_type, text_fields in sources:
            source_path = self.data_dir / source_name
            
            if not source_path.exists():
                logger.warning(f"Source not found: {source_path}")
                continue
            
            logger.info(f"Loading from {source_name} (type: {source_type})")
            texts_from_source = []
            
            try:
                if source_type == "jsonl":
                    text_iterator = self.load_jsonl(source_path, text_fields)
                elif source_type == "compressed":
                    text_iterator = self.load_compressed_text(source_path)
                elif source_type == "wiki":
                    text_iterator = self.load_wiki_dump(source_path)
                else:
                    continue
                
                # Collect texts with limit
                for i, text in enumerate(text_iterator):
                    if i >= max_texts_per_source:
                        break
                    texts_from_source.append(text)
                
                logger.info(f"Loaded {len(texts_from_source)} texts from {source_name}")
                all_texts.extend(texts_from_source)
                
            except Exception as e:
                logger.error(f"Error processing {source_name}: {e}")
        
        logger.info(f"Total texts loaded: {len(all_texts)}")
        return all_texts


def train_tokenizer():
    """Main function to train the tokenizer"""
    logger.info("Starting comprehensive tokenizer training...")
    
    # Create dataset loader
    loader = DatasetLoader()
    
    # Load all training texts
    logger.info("Loading training corpus...")
    training_texts = loader.get_all_training_texts(max_texts_per_source=50000)
    
    if not training_texts:
        logger.error("No training texts loaded! Please check your dataset directory.")
        return
    
    logger.info(f"Loaded {len(training_texts)} texts for training")
    
    # Create tokenizer configuration
    config = TokenizerConfig(
        vocab_size=32000,      # Good size for SLM
        min_frequency=5,       # Filter rare tokens
        max_token_length=16,   # Handle long Indic words
        special_tokens=[
            "<pad>", "<unk>", "<s>", "</s>", "<mask>",
            "<sep>", "<cls>", 
            "<te>", "<hi>", "<en>",  # Language tokens
            "<sum>", "<qa>", "<trans>",  # Task tokens
        ]
    )
    
    # Initialize tokenizer
    tokenizer = IndicTokenizer(config)
    
    # Train tokenizer
    logger.info("Training tokenizer...")
    tokenizer.train(training_texts)
    
    # Create models directory for tokenizer
    tokenizer_dir = Path("models/indic_tokenizer")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer.save(str(tokenizer_dir))
    
    # Test tokenizer
    logger.info("Testing tokenizer...")
    test_texts = [
        "ఇది ఒక తెలుగు వాక్యం.",
        "यह एक हिंदी वाक्य है।",
        "This is an English sentence.",
        "Mixed తెలుగు और हिंदी text example.",
        "Advanced test with conjuncts: క్ష్మ, क्ष्म, and punctuation।",
        "Longer text for testing: ఆంధ్రప్రదేశ్ రాష్ట్రంలో తెలుగు భాష మాట్లాడుతారు। भारत में हिंदी भाषा बोली जाती है।"
    ]
    
    logger.info("\n" + "="*60)
    logger.info("TOKENIZER TEST RESULTS")
    logger.info("="*60)
    
    for i, test_text in enumerate(test_texts, 1):
        logger.info(f"\nTest {i}: {test_text}")
        
        # Tokenize
        tokens = tokenizer.tokenize(test_text)
        logger.info(f"Tokens: {tokens}")
        
        # Encode
        encoded = tokenizer.encode(test_text)
        logger.info(f"Encoded: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        
        # Decode
        decoded = tokenizer.decode(encoded)
        logger.info(f"Decoded: {decoded}")
        
        # Check round-trip accuracy
        # Remove language and special tokens for comparison
        original_clean = test_text.strip()
        decoded_clean = decoded.replace("<te>", "").replace("<hi>", "").replace("<en>", "").strip()
        
        if original_clean.replace(" ", "") == decoded_clean.replace(" ", ""):
            logger.info("✅ Round-trip: PERFECT")
        else:
            logger.info("⚠️ Round-trip: APPROXIMATE")
    
    # Print tokenizer statistics
    logger.info("\n" + "="*60)
    logger.info("TOKENIZER STATISTICS")
    logger.info("="*60)
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    logger.info(f"Number of merge rules: {len(tokenizer.merge_rules)}")
    logger.info(f"Special tokens: {tokenizer.get_special_tokens_dict()}")
    
    # Sample vocabulary
    logger.info("\nSample vocabulary (first 50 tokens):")
    for i in range(min(50, tokenizer.get_vocab_size())):
        if i in tokenizer.id_to_token:
            token = tokenizer.id_to_token[i]
            logger.info(f"  {i:4d}: '{token}'")
    
    logger.info(f"\nTokenizer successfully trained and saved to: {tokenizer_dir}")
    logger.info("You can now use this tokenizer in your SLM model!")
    
    return tokenizer


def test_existing_tokenizer():
    """Test an existing tokenizer if available"""
    tokenizer_dir = Path("models/indic_tokenizer")
    
    if not tokenizer_dir.exists():
        logger.info("No existing tokenizer found. Please train one first.")
        return
    
    logger.info("Loading existing tokenizer...")
    try:
        tokenizer = IndicTokenizer.load(str(tokenizer_dir))
        logger.info(f"Loaded tokenizer with {tokenizer.get_vocab_size()} tokens")
        
        # Quick test
        test_text = "తెలుగు మరియు हिंदी mixed text test।"
        tokens = tokenizer.tokenize(test_text)
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        logger.info(f"Test text: {test_text}")
        logger.info(f"Tokens: {tokens}")
        logger.info(f"Encoded: {encoded}")
        logger.info(f"Decoded: {decoded}")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test Indic tokenizer")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                      help="Mode: train new tokenizer or test existing one")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_tokenizer()
    elif args.mode == "test":
        test_existing_tokenizer()
