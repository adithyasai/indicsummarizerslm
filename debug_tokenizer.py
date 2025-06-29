#!/usr/bin/env python3
"""
Debug IndicBART tokenizer to understand language tokens
"""

from transformers import AutoTokenizer
import torch

def debug_indicbart_tokenizer():
    print("🔍 Debugging IndicBART Tokenizer")
    print("=" * 40)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print()
    
    # Check special tokens
    print("Special tokens:")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print()
    
    # Look for language tokens
    print("Looking for language tokens:")
    possible_tokens = ['<2te>', '<te>', '<2hi>', '<hi>', 'te_IN', 'hi_IN', '__te__', '__hi__']
    
    for token in possible_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"   {token}: ID {token_id}")
        except:
            print(f"   {token}: NOT FOUND")
    
    print()
    
    # Test simple tokenization
    telugu_text = "హైదరాబాద్‌లో కృత్రిమ మేధతకు సంబంధించిన కొత్త పరిశోధనలు"
    
    print("Testing tokenization:")
    print(f"Input: {telugu_text}")
    
    # Try different approaches
    approaches = [
        ("Plain text", telugu_text),
        ("With <2te>", f"<2te> {telugu_text}"),
        ("With te_IN:", f"te_IN: {telugu_text}"),
        ("Summarize prefix", f"summarize: {telugu_text}")
    ]
    
    for name, text in approaches:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        print(f"\n{name}:")
        print(f"   Tokens: {tokens[:10]}...")  # First 10 tokens
        print(f"   IDs: {token_ids[:10]}...")
        
        # Try encoding/decoding
        encoded = tokenizer.encode(text, return_tensors="pt", max_length=50, truncation=True)
        decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
        print(f"   Decoded: {decoded}")

if __name__ == "__main__":
    debug_indicbart_tokenizer()
