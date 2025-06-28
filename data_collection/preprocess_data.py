#!/usr/bin/env py
"""
Data Preprocessing Pipeline for AadiShakthiSLM
Cleans, filters, and prepares Telugu and Hindi text data for training
"""

import os
import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import logging
from collections import defaultdict, Counter
import unicodedata
import concurrent.futures
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocesses Telugu and Hindi text data for SLM training"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Language-specific configurations
        self.lang_configs = {
            "te": {
                "name": "Telugu",
                "unicode_range": (0x0C00, 0x0C7F),  # Telugu Unicode block
                "common_chars": "అఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహళఱ",
                "numerals": "౦౧౨౩౪౫౬౭౮౯",
                "punctuation": "।॥‌‍"  # Devanagari punctuation
            },
            "hi": {
                "name": "Hindi",
                "unicode_range": (0x0900, 0x097F),  # Devanagari Unicode block
                "common_chars": "अआइईउऊऋऌएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
                "numerals": "०१२३४५६७८९",
                "punctuation": "।॥‌‍"
            }
        }
        
        # Text quality filters
        self.quality_filters = {
            "min_length": 50,        # Minimum characters
            "max_length": 2000,      # Maximum characters  
            "min_words": 5,          # Minimum words
            "max_words": 400,        # Maximum words
            "min_sentences": 2,      # Minimum sentences
            "max_repetition": 0.3,   # Max ratio of repeated n-grams
            "min_lang_ratio": 0.7,   # Min ratio of target language characters
            "max_digits": 0.2,       # Max ratio of digits
            "max_punctuation": 0.1   # Max ratio of punctuation
        }

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language and confidence score"""
        if not text.strip():
            return "unknown", 0.0
        
        lang_scores = {}
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return "unknown", 0.0
        
        for lang, config in self.lang_configs.items():
            start, end = config["unicode_range"]
            lang_chars = sum(1 for c in text if start <= ord(c) <= end)
            lang_scores[lang] = lang_chars / total_chars
        
        if not lang_scores:
            return "unknown", 0.0
        
        best_lang = max(lang_scores, key=lang_scores.get)
        confidence = lang_scores[best_lang]
        
        return best_lang, confidence

    def clean_text(self, text: str, language: str = None) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep language-specific punctuation
        if language in self.lang_configs:
            allowed_punct = self.lang_configs[language]["punctuation"] + ".,!?;:'\"-()[]"
            # Keep only letters, digits, whitespace, and allowed punctuation
            pattern = f"[^\\w\\s{re.escape(allowed_punct)}]"
            text = re.sub(pattern, '', text)
        
        # Clean up whitespace again
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def calculate_text_quality(self, text: str, language: str) -> Dict[str, float]:
        """Calculate various text quality metrics"""
        if not text:
            return {"valid": False, "score": 0.0}
        
        words = text.split()
        sentences = re.split(r'[।.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        metrics = {
            "length": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0
        }
        
        # Language ratio
        if language in self.lang_configs:
            start, end = self.lang_configs[language]["unicode_range"]
            lang_chars = sum(1 for c in text if start <= ord(c) <= end)
            total_alpha = sum(1 for c in text if c.isalpha())
            metrics["lang_ratio"] = lang_chars / total_alpha if total_alpha > 0 else 0
        else:
            metrics["lang_ratio"] = 0
        
        # Digit ratio
        digit_chars = sum(1 for c in text if c.isdigit())
        metrics["digit_ratio"] = digit_chars / len(text) if text else 0
        
        # Punctuation ratio
        punct_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        metrics["punct_ratio"] = punct_chars / len(text) if text else 0
        
        # Repetition check (simple n-gram repetition)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        if bigrams:
            bigram_counts = Counter(bigrams)
            most_common_count = bigram_counts.most_common(1)[0][1] if bigram_counts else 0
            metrics["repetition_ratio"] = most_common_count / len(bigrams)
        else:
            metrics["repetition_ratio"] = 0
        
        # Quality score
        quality_score = 1.0
        filters = self.quality_filters
        
        # Length checks
        if metrics["length"] < filters["min_length"] or metrics["length"] > filters["max_length"]:
            quality_score *= 0.5
        
        if metrics["word_count"] < filters["min_words"] or metrics["word_count"] > filters["max_words"]:
            quality_score *= 0.5
        
        if metrics["sentence_count"] < filters["min_sentences"]:
            quality_score *= 0.3
        
        # Quality checks
        if metrics["lang_ratio"] < filters["min_lang_ratio"]:
            quality_score *= 0.2
        
        if metrics["digit_ratio"] > filters["max_digits"]:
            quality_score *= 0.7
        
        if metrics["punct_ratio"] > filters["max_punctuation"]:
            quality_score *= 0.8
        
        if metrics["repetition_ratio"] > filters["max_repetition"]:
            quality_score *= 0.6
        
        metrics["valid"] = quality_score > 0.5
        metrics["score"] = quality_score
        
        return metrics

    def process_text_file(self, file_path: Path, output_file: Path, language: str = None) -> Dict:
        """Process a single text file"""
        stats = {
            "input_lines": 0,
            "output_lines": 0,
            "languages": defaultdict(int),
            "quality_scores": []
        }
        
        logger.info(f"Processing {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile, \
                 open(output_file, 'w', encoding='utf-8') as outfile:
                
                for line_num, line in enumerate(infile, 1):
                    stats["input_lines"] += 1
                    
                    # Progress indicator
                    if line_num % 10000 == 0:
                        logger.info(f"Processed {line_num} lines from {file_path.name}")
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Detect language if not specified
                    if language is None:
                        detected_lang, confidence = self.detect_language(line)
                        if confidence < 0.5:
                            continue
                        current_lang = detected_lang
                    else:
                        current_lang = language
                    
                    stats["languages"][current_lang] += 1
                    
                    # Skip if not Telugu or Hindi
                    if current_lang not in ["te", "hi"]:
                        continue
                    
                    # Clean text
                    cleaned = self.clean_text(line, current_lang)
                    if not cleaned:
                        continue
                    
                    # Check quality
                    quality = self.calculate_text_quality(cleaned, current_lang)
                    stats["quality_scores"].append(quality["score"])
                    
                    if not quality["valid"]:
                        continue
                    
                    # Write to output
                    output_data = {
                        "text": cleaned,
                        "language": current_lang,
                        "quality_score": quality["score"],
                        "word_count": quality["word_count"],
                        "source_file": file_path.name
                    }
                    
                    outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    stats["output_lines"] += 1
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        # Calculate average quality score
        if stats["quality_scores"]:
            stats["avg_quality"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
        else:
            stats["avg_quality"] = 0.0
        
        return stats

    def create_summarization_dataset(self, input_dir: Path, output_file: Path) -> Dict:
        """Create dataset specifically for summarization training"""
        stats = {
            "total_pairs": 0,
            "valid_pairs": 0,
            "languages": defaultdict(int)
        }
        
        logger.info(f"Creating summarization dataset from {input_dir}")
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # Process XLSum-style files
            for json_file in input_dir.rglob("*.json*"):
                logger.info(f"Processing summarization file: {json_file}")
                
                try:
                    # Handle different file formats
                    if json_file.suffix == '.jsonl':
                        # JSONL format
                        with open(json_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    data = json.loads(line)
                                    if self.process_summary_pair(data, outfile, stats):
                                        stats["total_pairs"] += 1
                    else:
                        # JSON format
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if self.process_summary_pair(item, outfile, stats):
                                        stats["total_pairs"] += 1
                            elif isinstance(data, dict):
                                if self.process_summary_pair(data, outfile, stats):
                                    stats["total_pairs"] += 1
                
                except Exception as e:
                    logger.warning(f"Could not process {json_file}: {e}")
        
        return stats

    def process_summary_pair(self, data: Dict, outfile, stats: Dict) -> bool:
        """Process a single text-summary pair"""
        try:
            # Extract text and summary from various formats
            text = None
            summary = None
            
            # Common field names for text
            for field in ["text", "document", "article", "content", "body"]:
                if field in data and data[field]:
                    text = str(data[field]).strip()
                    break
            
            # Common field names for summary
            for field in ["summary", "title", "headline", "abstract"]:
                if field in data and data[field]:
                    summary = str(data[field]).strip()
                    break
            
            if not text or not summary:
                return False
            
            # Detect language
            text_lang, text_conf = self.detect_language(text)
            summary_lang, summary_conf = self.detect_language(summary)
            
            # Only keep Telugu and Hindi
            if text_lang not in ["te", "hi"] or summary_lang not in ["te", "hi"]:
                return False
            
            # Languages should match
            if text_lang != summary_lang:
                return False
            
            language = text_lang
            
            # Clean text
            clean_text = self.clean_text(text, language)
            clean_summary = self.clean_text(summary, language)
            
            if not clean_text or not clean_summary:
                return False
            
            # Quality checks
            text_quality = self.calculate_text_quality(clean_text, language)
            summary_quality = self.calculate_text_quality(clean_summary, language)
            
            # More lenient quality for summaries
            if not text_quality["valid"] or summary_quality["score"] < 0.3:
                return False
            
            # Length checks for summarization
            if len(clean_summary.split()) >= len(clean_text.split()):
                return False  # Summary should be shorter
            
            if len(clean_text.split()) < 20:  # Text too short to summarize
                return False
            
            # Create training example
            example = {
                "input_text": clean_text,
                "target_summary": clean_summary,
                "language": language,
                "input_length": len(clean_text.split()),
                "summary_length": len(clean_summary.split()),
                "compression_ratio": len(clean_summary.split()) / len(clean_text.split()),
                "quality_score": (text_quality["score"] + summary_quality["score"]) / 2
            }
            
            outfile.write(json.dumps(example, ensure_ascii=False) + '\n')
            stats["valid_pairs"] += 1
            stats["languages"][language] += 1
            
            return True
            
        except Exception as e:
            logger.warning(f"Error processing summary pair: {e}")
            return False

    def process_all_files(self, max_workers: int = 4) -> Dict:
        """Process all files in the raw directory"""
        results = {
            "processed_files": [],
            "total_stats": {
                "input_lines": 0,
                "output_lines": 0,
                "languages": defaultdict(int),
                "avg_quality": 0.0
            }
        }
        
        # Find all text files to process
        text_files = []
        for pattern in ["*.txt", "*.json", "*.jsonl"]:
            text_files.extend(self.raw_dir.rglob(pattern))
        
        logger.info(f"Found {len(text_files)} files to process")
        
        # Process files
        for file_path in text_files:
            if file_path.stat().st_size == 0:
                continue
                
            output_name = f"{file_path.stem}_processed.jsonl"
            output_path = self.processed_dir / output_name
            
            # Skip if already processed
            if output_path.exists():
                logger.info(f"Skipping {file_path.name} (already processed)")
                continue
            
            stats = self.process_text_file(file_path, output_path)
            results["processed_files"].append({
                "input_file": str(file_path),
                "output_file": str(output_path),
                "stats": stats
            })
            
            # Update total stats
            results["total_stats"]["input_lines"] += stats["input_lines"]
            results["total_stats"]["output_lines"] += stats["output_lines"]
            for lang, count in stats["languages"].items():
                results["total_stats"]["languages"][lang] += count
        
        # Create summarization dataset
        summary_output = self.processed_dir / "summarization_dataset.jsonl"
        if not summary_output.exists():
            summary_stats = self.create_summarization_dataset(self.raw_dir, summary_output)
            results["summarization_stats"] = summary_stats
        
        # Save processing report
        report_path = self.processed_dir / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Processing complete. Report saved to {report_path}")
        return results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess text data for AadiShakthiSLM")
    parser.add_argument("--data-dir", default="data", help="Base directory for datasets")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--language", choices=["te", "hi"], help="Process only specific language")
    
    args = parser.parse_args()
    
    preprocessor = TextPreprocessor(args.data_dir)
    results = preprocessor.process_all_files(max_workers=args.workers)
    
    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING RESULTS")
    print("="*50)
    
    total_stats = results["total_stats"]
    print(f"Input lines: {total_stats['input_lines']:,}")
    print(f"Output lines: {total_stats['output_lines']:,}")
    print(f"Retention rate: {total_stats['output_lines']/total_stats['input_lines']*100:.1f}%")
    
    print("\nLanguage distribution:")
    for lang, count in total_stats["languages"].items():
        print(f"  {lang}: {count:,} lines")
    
    if "summarization_stats" in results:
        summary_stats = results["summarization_stats"]
        print(f"\nSummarization pairs: {summary_stats['valid_pairs']:,}")

if __name__ == "__main__":
    main()
