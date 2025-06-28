#!/usr/bin/env py
"""
Custom Tokenizer for AadiShakthiSLM - Telugu and Hindi SLM
Handles Indic script characteristics, conjunct consonants, and multilingual text
"""

import os
import json
import pickle
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set, Union
from pathlib import Path
import regex as re
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TokenizerConfig:
    """Configuration for the Indic tokenizer"""
    vocab_size: int = 50000
    min_frequency: int = 2
    max_token_length: int = 16
    special_tokens: List[str] = None
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    mask_token: str = "<mask>"
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                self.pad_token, self.unk_token, self.bos_token, 
                self.eos_token, self.mask_token,
                "<sep>", "<cls>", "<te>", "<hi>", "<en>"  # Language tokens
            ]

class IndicTokenizer:
    """
    Custom tokenizer for Telugu and Hindi text with support for:
    - Indic script characteristics
    - Conjunct consonants
    - Multilingual text
    - Subword tokenization
    """
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab = {}
        self.id_to_token = {}
        self.token_to_id = {}
        self.merge_rules = []
        self.language_patterns = self._init_language_patterns()
        self.unicode_categories = self._init_unicode_categories()
        
        # Initialize special tokens
        self._init_special_tokens()
        
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for i, token in enumerate(self.config.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
    def _init_language_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for different languages"""
        return {
            'telugu': re.compile(r'[\u0C00-\u0C7F]+'),
            'hindi': re.compile(r'[\u0900-\u097F]+'),
            'english': re.compile(r'[a-zA-Z]+'),
            'numbers': re.compile(r'[\d]+'),
            'punctuation': re.compile(r'[^\w\s]'),
            # Indic script specific patterns
            'telugu_conjuncts': re.compile(r'[\u0C15-\u0C39][\u0C4D][\u0C15-\u0C39]'),
            'hindi_conjuncts': re.compile(r'[\u0915-\u0939][\u094D][\u0915-\u0939]'),
            'telugu_vowels': re.compile(r'[\u0C05-\u0C14\u0C3E-\u0C4C\u0C55\u0C56]'),
            'hindi_vowels': re.compile(r'[\u0905-\u0914\u093E-\u094C\u0955\u0956]'),
        }
    
    def _init_unicode_categories(self) -> Dict[str, Set[str]]:
        """Initialize Unicode category mappings for Indic scripts"""
        return {
            'telugu_consonants': set(chr(i) for i in range(0x0C15, 0x0C39 + 1)),
            'telugu_vowels': set(chr(i) for i in range(0x0C05, 0x0C14 + 1)),
            'telugu_matras': set(chr(i) for i in range(0x0C3E, 0x0C4C + 1)),
            'hindi_consonants': set(chr(i) for i in range(0x0915, 0x0939 + 1)),
            'hindi_vowels': set(chr(i) for i in range(0x0905, 0x0914 + 1)),
            'hindi_matras': set(chr(i) for i in range(0x093E, 0x094C + 1)),
            'virama': {'\u0C4D', '\u094D'},  # Telugu and Hindi virama
        }
    
    def detect_language(self, text: str) -> str:
        """Detect primary language of text"""
        text_len = len(text)
        if text_len == 0:
            return "unknown"
            
        te_count = len(self.language_patterns['telugu'].findall(text))
        hi_count = len(self.language_patterns['hindi'].findall(text))
        en_count = len(self.language_patterns['english'].findall(text))
        
        total_chars = te_count + hi_count + en_count
        if total_chars == 0:
            return "unknown"
            
        if te_count > hi_count and te_count > en_count:
            return "telugu"
        elif hi_count > te_count and hi_count > en_count:
            return "hindi"
        elif en_count > 0:
            return "english"
        else:
            return "mixed"
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent tokenization"""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle punctuation spacing
        text = re.sub(r'([।॥।])', r' \1 ', text)  # Indic punctuation
        text = re.sub(r'([.!?])', r' \1 ', text)   # Western punctuation
        
        return text.strip()
    
    def pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenization step to handle script boundaries and word boundaries"""
        normalized_text = self.normalize_text(text)
        
        # Split by script boundaries and spaces
        tokens = []
        current_token = ""
        current_script = None
        
        for char in normalized_text:
            char_script = self._get_char_script(char)
            
            if char_script != current_script and current_token:
                tokens.append(current_token)
                current_token = ""
            
            if char == ' ':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                current_script = None
            else:
                current_token += char
                current_script = char_script
        
        if current_token:
            tokens.append(current_token)
            
        return [token for token in tokens if token.strip()]
    
    def _get_char_script(self, char: str) -> str:
        """Determine script of a character"""
        code_point = ord(char)
        
        if 0x0C00 <= code_point <= 0x0C7F:
            return "telugu"
        elif 0x0900 <= code_point <= 0x097F:
            return "hindi"
        elif char.isalpha() and ord(char) < 256:
            return "english"
        elif char.isdigit():
            return "number"
        else:
            return "other"
    
    def handle_conjuncts(self, text: str) -> List[str]:
        """Handle Indic conjunct consonants as single units"""
        tokens = []
        i = 0
        
        while i < len(text):
            # Check for conjunct patterns (consonant + virama + consonant)
            if i < len(text) - 2:
                if (text[i] in self.unicode_categories['telugu_consonants'] and 
                    text[i + 1] == '\u0C4D' and 
                    text[i + 2] in self.unicode_categories['telugu_consonants']):
                    tokens.append(text[i:i+3])
                    i += 3
                    continue
                elif (text[i] in self.unicode_categories['hindi_consonants'] and 
                      text[i + 1] == '\u094D' and 
                      text[i + 2] in self.unicode_categories['hindi_consonants']):
                    tokens.append(text[i:i+3])
                    i += 3
                    continue
            
            tokens.append(text[i])
            i += 1
            
        return tokens
    
    def collect_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Collect vocabulary from training texts"""
        vocab_counter = Counter()
        
        logger.info(f"Collecting vocabulary from {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            if i % 10000 == 0:
                logger.info(f"Processed {i} texts...")
                
            # Pre-tokenize
            pre_tokens = self.pre_tokenize(text)
            
            for token in pre_tokens:
                # Handle conjuncts
                conjunct_tokens = self.handle_conjuncts(token)
                
                # Add individual characters
                for char_token in conjunct_tokens:
                    vocab_counter[char_token] += 1
                
                # Add subword units (for BPE-like behavior)
                for length in range(2, min(len(token) + 1, self.config.max_token_length + 1)):
                    for start in range(len(token) - length + 1):
                        subword = token[start:start + length]
                        vocab_counter[subword] += 1
        
        # Filter by frequency
        filtered_vocab = {
            token: count for token, count in vocab_counter.items()
            if count >= self.config.min_frequency
        }
        
        logger.info(f"Collected {len(filtered_vocab)} vocabulary items")
        return filtered_vocab
    
    def build_bpe_merges(self, vocab: Dict[str, int]) -> List[Tuple[str, str]]:
        """Build BPE merge rules for subword tokenization"""
        logger.info("Building BPE merge rules...")
        
        # Start with character-level vocabulary
        working_vocab = {char: count for char, count in vocab.items() if len(char) == 1}
        merge_rules = []
        
        for merge_step in range(self.config.vocab_size - len(self.config.special_tokens)):
            if merge_step % 1000 == 0:
                logger.info(f"BPE merge step {merge_step}")
                
            # Find most frequent pair
            pair_counts = Counter()
            
            for word, count in working_vocab.items():
                if len(word) < 2:
                    continue
                    
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] += count
            
            if not pair_counts:
                break
                
            best_pair = pair_counts.most_common(1)[0][0]
            merge_rules.append(best_pair)
            
            # Apply merge
            new_vocab = {}
            for word, count in working_vocab.items():
                new_word = word.replace(best_pair[0] + best_pair[1], 
                                      best_pair[0] + best_pair[1])
                new_vocab[new_word] = count
            
            working_vocab = new_vocab
        
        logger.info(f"Built {len(merge_rules)} BPE merge rules")
        return merge_rules
    
    def finalize_vocab(self, vocab: Dict[str, int]) -> None:
        """Finalize vocabulary with most frequent tokens"""
        # Sort by frequency
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        
        # Reserve space for special tokens
        available_slots = self.config.vocab_size - len(self.config.special_tokens)
        
        # Add most frequent tokens
        current_id = len(self.config.special_tokens)
        
        for token, count in sorted_vocab[:available_slots]:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        logger.info(f"Finalized vocabulary with {len(self.token_to_id)} tokens")
    
    def train(self, texts: List[str]) -> None:
        """Train the tokenizer on a corpus"""
        logger.info("Starting tokenizer training...")
        
        # Collect vocabulary
        vocab = self.collect_vocab(texts)
        
        # Build merge rules
        self.merge_rules = self.build_bpe_merges(vocab)
        
        # Finalize vocabulary
        self.finalize_vocab(vocab)
        
        logger.info("Tokenizer training completed!")
    
    def apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word"""
        if len(word) <= 1:
            return [word]
            
        # Start with character-level
        word_tokens = list(word)
        
        # Apply merge rules
        for merge_rule in self.merge_rules:
            new_tokens = []
            i = 0
            
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and 
                    word_tokens[i] == merge_rule[0] and 
                    word_tokens[i + 1] == merge_rule[1]):
                    new_tokens.append(merge_rule[0] + merge_rule[1])
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            
            word_tokens = new_tokens
            
        return word_tokens
    
    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[str]:
        """Tokenize text into subwords with enhanced fallback handling"""
        tokens = []
        
        if add_special_tokens:
            lang = self.detect_language(text)
            if lang in ['telugu', 'hindi', 'english']:
                tokens.append(f"<{lang[:2]}>")
            tokens.append(self.config.bos_token)
        
        # Pre-tokenize
        pre_tokens = self.pre_tokenize(text)
        
        for pre_token in pre_tokens:
            # Apply BPE
            bpe_tokens = self.apply_bpe(pre_token)
            
            for token in bpe_tokens:
                if token in self.token_to_id:
                    tokens.append(token)
                else:
                    # Enhanced fallback chain
                    token_added = False
                    
                    # 1. Try subword decomposition
                    if len(token) > 1:
                        subwords = self._decompose_to_subwords(token)
                        if subwords:
                            tokens.extend(subwords)
                            token_added = True
                    
                    # 2. Try n-gram matching if subword failed
                    if not token_added and len(token) > 2:
                        ngrams = self._find_matching_ngrams(token)
                        if ngrams:
                            tokens.extend(ngrams)
                            token_added = True
                    
                    # 3. Try unicode category fallback
                    if not token_added:
                        category_tokens = self._tokenize_by_category(token)
                        if category_tokens:
                            tokens.extend(category_tokens)
                            token_added = True
                    
                    # 4. Final character-level fallback
                    if not token_added:
                        for char in token:
                            if char in self.token_to_id:
                                tokens.append(char)
                            else:
                                tokens.append(self.config.unk_token)
        
        if add_special_tokens:
            tokens.append(self.config.eos_token)
            
        return tokens

    def _decompose_to_subwords(self, token: str) -> List[str]:
        """Try to decompose token into known subwords"""
        best_subwords = []
        min_unk = float('inf')
        
        # Try different splits to minimize unknown tokens
        for i in range(1, len(token)):
            left = token[:i]
            right = token[i:]
            
            left_tokens = []
            if left in self.token_to_id:
                left_tokens = [left]
            else:
                for c in left:
                    left_tokens.append(c if c in self.token_to_id else self.config.unk_token)
                    
            right_tokens = []
            if right in self.token_to_id:
                right_tokens = [right]
            else:
                for c in right:
                    right_tokens.append(c if c in self.token_to_id else self.config.unk_token)
            
            combined = left_tokens + right_tokens
            unk_count = combined.count(self.config.unk_token)
            
            if unk_count < min_unk:
                min_unk = unk_count
                best_subwords = combined
        
        return best_subwords if best_subwords else []

    def _find_matching_ngrams(self, token: str) -> List[str]:
        """Find matching n-grams in vocabulary"""
        matches = []
        n = len(token)
        
        while n > 1:
            found = False
            for i in range(len(token) - n + 1):
                ngram = token[i:i+n]
                if ngram in self.token_to_id:
                    matches.append(ngram)
                    found = True
            if found:
                break
            n -= 1
            
        return matches

    def _tokenize_by_category(self, token: str) -> List[str]:
        """Tokenize based on Unicode categories"""
        category_tokens = []
        current_token = ""
        current_category = None
        
        for char in token:
            char_category = unicodedata.category(char)[0]
            
            if char_category != current_category and current_token:
                if current_token in self.token_to_id:
                    category_tokens.append(current_token)
                else:
                    # Split by each character in the sequence
                    for c in current_token:
                        category_tokens.append(c if c in self.token_to_id else self.config.unk_token)
                current_token = ""
            
            current_token += char
            current_category = char_category
        
        # Handle last token
        if current_token:
            if current_token in self.token_to_id:
                category_tokens.append(current_token)
            else:
                for c in current_token:
                    category_tokens.append(c if c in self.token_to_id else self.config.unk_token)
                    
        return category_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text, add_special_tokens)
        return [self.token_to_id.get(token, self.token_to_id[self.config.unk_token]) 
                for token in tokens]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text with improved Telugu word boundary heuristic"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.config.special_tokens:
                    continue
                tokens.append(token)
        # Heuristic: add a space after tokens that end with a Telugu vowel, punctuation, or whitespace
        text = ''
        for i, token in enumerate(tokens):
            text += token
            # Telugu vowels and common punctuation unicode ranges
            if re.search(r'[\u0C05-\u0C14\u0C3E-\u0C4C\u0C55\u0C56।॥.!?\s]$', token):
                text += ' '
        # Clean up double spaces and fix punctuation spacing
        text = re.sub(r' +', ' ', text)
        text = re.sub(r' ([।॥।.!?])', r'\1', text)
        return text.strip()
    
    def save(self, save_directory: str) -> None:
        """Save tokenizer to directory"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = save_path / "tokenizer_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.config.vocab_size,
                'min_frequency': self.config.min_frequency,
                'max_token_length': self.config.max_token_length,
                'special_tokens': self.config.special_tokens,
                'unk_token': self.config.unk_token,
                'pad_token': self.config.pad_token,
                'bos_token': self.config.bos_token,
                'eos_token': self.config.eos_token,
                'mask_token': self.config.mask_token,
            }, f, ensure_ascii=False, indent=2)
        
        # Save vocabulary
        vocab_path = save_path / "vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        
        # Save merge rules
        merges_path = save_path / "merges.txt"
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for merge_rule in self.merge_rules:
                f.write(f"{merge_rule[0]} {merge_rule[1]}\n")
        
        logger.info(f"Tokenizer saved to {save_directory}")
    
    @classmethod
    def load(cls, load_directory: str) -> 'IndicTokenizer':
        """Load tokenizer from directory, filtering out unknown config keys"""
        load_path = Path(load_directory)
        config_path = load_path / "tokenizer_config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        # Only keep keys that TokenizerConfig accepts
        valid_keys = {f.name for f in TokenizerConfig.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_data.items() if k in valid_keys}
        config = TokenizerConfig(**filtered_config)
        tokenizer = cls(config)
        
        # Load vocabulary
        vocab_path = load_path / "vocab.json"
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer.token_to_id = json.load(f)
        
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        
        # Load merge rules
        merges_path = load_path / "merges.txt"
        if merges_path.exists():
            tokenizer.merge_rules = []
            with open(merges_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip version line
                for line in lines:
                    if line.strip():
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            tokenizer.merge_rules.append((parts[0], parts[1]))
        
        logger.info(f"Tokenizer loaded from {load_directory}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.token_to_id)
    
    def get_special_tokens_dict(self) -> Dict[str, int]:
        """Get special tokens mapping"""
        return {token: self.token_to_id[token] 
                for token in self.config.special_tokens 
                if token in self.token_to_id}
    
    @property
    def pad_token_id(self):
        """Return the ID of the pad token"""
        return self.token_to_id.get(self.config.pad_token, 0)


def main():
    """Example usage of the tokenizer"""
    # Sample texts for training
    sample_texts = [
        "ఇది తెలుగు వాక్యం. This is a Telugu sentence.",
        "यह हिंदी वाक्य है। This is a Hindi sentence.",
        "Mixed language text with తెలుగు and हिंदी words.",
        "Advanced tokenization test with conjuncts: క్ష్మ, क्ष्म",
    ]
    
    # Create and train tokenizer
    config = TokenizerConfig(vocab_size=1000)  # Small for demo
    tokenizer = IndicTokenizer(config)
    
    # Train on sample texts
    tokenizer.train(sample_texts * 100)  # Repeat for frequency
    
    # Test tokenization
    test_text = "ఇది టెస్ట్ వాక్యం है।"
    tokens = tokenizer.tokenize(test_text)
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Save tokenizer
    tokenizer.save("models/tokenizer")
    
    # Load tokenizer
    loaded_tokenizer = IndicTokenizer.load("models/tokenizer")
    print(f"Loaded tokenizer vocab size: {loaded_tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
