#!/usr/bin/env py
"""
Training Data Pipeline for AadiShakthiSLM
Creates training, validation, and test datasets for the model
"""

import os
import json
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import logging
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDataPipeline:
    """Creates and manages training datasets for AadiShakthiSLM"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.config = {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "max_sequence_length": 512,
            "min_sequence_length": 50,
            "max_samples_per_language": 100000,  # Limit for balanced training
            "random_seed": 42
        }
        
        # Set random seeds for reproducibility
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

    def load_processed_data(self) -> Dict[str, List[Dict]]:
        """Load all processed data files"""
        data = {
            "text_corpus": [],
            "summarization": []
        }
        
        logger.info("Loading processed data...")
        
        # Load text corpus data
        for jsonl_file in self.processed_dir.rglob("*_processed.jsonl"):
            logger.info(f"Loading {jsonl_file}")
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                item = json.loads(line)
                                data["text_corpus"].append(item)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON at {jsonl_file}:{line_num}: {e}")
                        
                        # Progress indicator for large files
                        if line_num % 50000 == 0:
                            logger.info(f"Loaded {line_num} lines from {jsonl_file.name}")
                            
            except Exception as e:
                logger.error(f"Error loading {jsonl_file}: {e}")
        
        # Load summarization data
        summary_file = self.processed_dir / "summarization_dataset.jsonl"
        if summary_file.exists():
            logger.info(f"Loading summarization data from {summary_file}")
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                data["summarization"].append(item)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON in summarization data: {e}")
            except Exception as e:
                logger.error(f"Error loading summarization data: {e}")
        
        logger.info(f"Loaded {len(data['text_corpus'])} text samples and {len(data['summarization'])} summarization pairs")
        return data

    def create_language_model_data(self, text_data: List[Dict]) -> Dict[str, List[str]]:
        """Create language modeling datasets (for pre-training)"""
        logger.info("Creating language modeling datasets...")
        
        language_data = defaultdict(list)
        
        for item in text_data:
            language = item.get("language", "unknown")
            text = item.get("text", "").strip()
            
            if not text or language not in ["te", "hi"]:
                continue
            
            # Split long texts into chunks
            words = text.split()
            if len(words) < self.config["min_sequence_length"] // 5:  # Rough word estimate
                continue
            
            # Create overlapping chunks
            chunk_size = self.config["max_sequence_length"] // 2  # Words per chunk
            overlap = chunk_size // 4  # 25% overlap
            
            for i in range(0, len(words) - chunk_size + 1, chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk) >= self.config["min_sequence_length"]:
                    language_data[language].append(chunk)
        
        # Balance languages
        min_samples = min(len(texts) for texts in language_data.values())
        balanced_samples = min(min_samples, self.config["max_samples_per_language"])
        
        for language in language_data:
            if len(language_data[language]) > balanced_samples:
                language_data[language] = random.sample(language_data[language], balanced_samples)
        
        logger.info(f"Created language modeling data:")
        for lang, texts in language_data.items():
            logger.info(f"  {lang}: {len(texts)} chunks")
        
        return dict(language_data)

    def create_summarization_data(self, summary_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Create summarization training datasets"""
        logger.info("Creating summarization datasets...")
        
        language_data = defaultdict(list)
        
        for item in summary_data:
            language = item.get("language", "unknown")
            input_text = item.get("input_text", "").strip()
            target_summary = item.get("target_summary", "").strip()
            
            if not input_text or not target_summary or language not in ["te", "hi"]:
                continue
            
            # Length filtering
            input_words = len(input_text.split())
            summary_words = len(target_summary.split())
            
            if (input_words < 20 or input_words > 400 or 
                summary_words < 5 or summary_words > 100):
                continue
            
            # Compression ratio check
            compression_ratio = summary_words / input_words
            if compression_ratio > 0.8 or compression_ratio < 0.1:
                continue
            
            training_example = {
                "input": input_text,
                "target": target_summary,
                "input_length": input_words,
                "target_length": summary_words,
                "compression_ratio": compression_ratio,
                "quality_score": item.get("quality_score", 0.5)
            }
            
            language_data[language].append(training_example)
        
        # Balance languages for summarization
        if len(language_data) > 1:
            min_samples = min(len(examples) for examples in language_data.values())
            for language in language_data:
                if len(language_data[language]) > min_samples:
                    # Sort by quality and take the best examples
                    sorted_examples = sorted(
                        language_data[language], 
                        key=lambda x: x.get("quality_score", 0), 
                        reverse=True
                    )
                    language_data[language] = sorted_examples[:min_samples]
        
        logger.info(f"Created summarization data:")
        for lang, examples in language_data.items():
            logger.info(f"  {lang}: {len(examples)} examples")
            if examples:
                avg_compression = np.mean([ex["compression_ratio"] for ex in examples])
                logger.info(f"    Average compression ratio: {avg_compression:.2f}")
        
        return dict(language_data)

    def split_dataset(self, data: List, dataset_type: str = "generic") -> Dict[str, List]:
        """Split dataset into train/val/test"""
        if not data:
            return {"train": [], "val": [], "test": []}
        
        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            data, 
            test_size=(self.config["val_ratio"] + self.config["test_ratio"]),
            random_state=self.config["random_seed"]
        )
        
        # Second split: val vs test
        if temp_data:
            val_ratio_adjusted = self.config["val_ratio"] / (self.config["val_ratio"] + self.config["test_ratio"])
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio_adjusted),
                random_state=self.config["random_seed"]
            )
        else:
            val_data, test_data = [], []
        
        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        
        logger.info(f"Split {dataset_type} dataset:")
        for split_name, split_data in splits.items():
            logger.info(f"  {split_name}: {len(split_data)} examples")
        
        return splits

    def save_language_model_datasets(self, language_data: Dict[str, List[str]]) -> None:
        """Save language modeling datasets"""
        lm_dir = self.output_dir / "language_modeling"
        lm_dir.mkdir(parents=True, exist_ok=True)
        
        for language, texts in language_data.items():
            logger.info(f"Saving language modeling data for {language}")
            
            # Split the data
            splits = self.split_dataset(texts, f"language_modeling_{language}")
            
            # Save each split
            for split_name, split_data in splits.items():
                output_file = lm_dir / f"{language}_{split_name}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for text in split_data:
                        f.write(text + '\n\n')  # Double newline to separate documents
                
                logger.info(f"Saved {len(split_data)} examples to {output_file}")

    def save_summarization_datasets(self, language_data: Dict[str, List[Dict]]) -> None:
        """Save summarization datasets"""
        sum_dir = self.output_dir / "summarization"
        sum_dir.mkdir(parents=True, exist_ok=True)
        
        for language, examples in language_data.items():
            logger.info(f"Saving summarization data for {language}")
            
            # Split the data
            splits = self.split_dataset(examples, f"summarization_{language}")
            
            # Save each split
            for split_name, split_data in splits.items():
                output_file = sum_dir / f"{language}_{split_name}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for example in split_data:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
                logger.info(f"Saved {len(split_data)} examples to {output_file}")

    def create_mixed_language_dataset(self, language_data: Dict[str, List]) -> List:
        """Create mixed-language dataset for multilingual training"""
        mixed_data = []
        
        # Combine all languages
        for language, data in language_data.items():
            for item in data:
                if isinstance(item, dict):
                    item["language"] = language
                    mixed_data.append(item)
                else:
                    mixed_data.append({"text": item, "language": language})
        
        # Shuffle to mix languages
        random.shuffle(mixed_data)
        
        logger.info(f"Created mixed-language dataset with {len(mixed_data)} examples")
        return mixed_data

    def generate_training_config(self) -> Dict:
        """Generate configuration file for training"""
        config = {
            "model_config": {
                "vocab_size": 50000,
                "hidden_size": 512,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 2048,
                "max_position_embeddings": self.config["max_sequence_length"],
                "dropout": 0.1,
                "layer_norm_eps": 1e-12
            },
            "training_config": {
                "batch_size": 8,
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "save_steps": 5000,
                "eval_steps": 1000,
                "logging_steps": 100
            },
            "data_config": {
                "max_sequence_length": self.config["max_sequence_length"],
                "train_ratio": self.config["train_ratio"],
                "val_ratio": self.config["val_ratio"],
                "test_ratio": self.config["test_ratio"]
            },
            "languages": ["te", "hi"],
            "tasks": ["language_modeling", "summarization"]
        }
        
        return config

    def create_all_datasets(self) -> Dict:
        """Main method to create all training datasets"""
        logger.info("Starting training data pipeline...")
        
        # Load processed data
        raw_data = self.load_processed_data()
        
        results = {
            "language_modeling": {},
            "summarization": {},
            "statistics": {}
        }
        
        # Create language modeling datasets
        if raw_data["text_corpus"]:
            lm_data = self.create_language_model_data(raw_data["text_corpus"])
            self.save_language_model_datasets(lm_data)
            results["language_modeling"] = {lang: len(texts) for lang, texts in lm_data.items()}
            
            # Create mixed-language dataset
            mixed_lm = self.create_mixed_language_dataset(lm_data)
            mixed_splits = self.split_dataset(mixed_lm, "mixed_language_modeling")
            
            # Save mixed dataset
            mixed_dir = self.output_dir / "mixed"
            mixed_dir.mkdir(parents=True, exist_ok=True)
            
            for split_name, split_data in mixed_splits.items():
                output_file = mixed_dir / f"mixed_lm_{split_name}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in split_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Create summarization datasets
        if raw_data["summarization"]:
            sum_data = self.create_summarization_data(raw_data["summarization"])
            self.save_summarization_datasets(sum_data)
            results["summarization"] = {lang: len(examples) for lang, examples in sum_data.items()}
            
            # Create mixed summarization dataset
            mixed_sum = self.create_mixed_language_dataset(sum_data)
            mixed_splits = self.split_dataset(mixed_sum, "mixed_summarization")
            
            for split_name, split_data in mixed_splits.items():
                output_file = mixed_dir / f"mixed_sum_{split_name}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in split_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Generate training configuration
        training_config = self.generate_training_config()
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved training configuration to {config_file}")
        
        # Save pipeline results
        results["statistics"] = {
            "total_lm_samples": sum(results["language_modeling"].values()),
            "total_sum_samples": sum(results["summarization"].values()),
            "config_file": str(config_file)
        }
        
        results_file = self.output_dir / "pipeline_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline completed. Results saved to {results_file}")
        return results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create training datasets for AadiShakthiSLM")
    parser.add_argument("--data-dir", default="data", help="Base directory for processed data")
    parser.add_argument("--output-dir", default="training_data", help="Output directory for training datasets")
    parser.add_argument("--max-samples", type=int, default=100000, help="Maximum samples per language")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    pipeline = TrainingDataPipeline(args.data_dir, args.output_dir)
    pipeline.config["max_samples_per_language"] = args.max_samples
    pipeline.config["random_seed"] = args.seed
    
    results = pipeline.create_all_datasets()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING DATA PIPELINE RESULTS")
    print("="*50)
    
    print("Language Modeling datasets:")
    for lang, count in results["language_modeling"].items():
        print(f"  {lang}: {count:,} samples")
    
    print("\nSummarization datasets:")
    for lang, count in results["summarization"].items():
        print(f"  {lang}: {count:,} samples")
    
    stats = results["statistics"]
    print(f"\nTotal samples:")
    print(f"  Language Modeling: {stats['total_lm_samples']:,}")
    print(f"  Summarization: {stats['total_sum_samples']:,}")
    
    print(f"\nConfiguration saved to: {stats['config_file']}")

if __name__ == "__main__":
    main()
