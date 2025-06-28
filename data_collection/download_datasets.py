#!/usr/bin/env py
"""
Data Collection Script for AadiShakthiSLM
Downloads and prepares Telugu and Hindi datasets for training
"""

import os
import json
import requests
import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from urllib.parse import urlparse
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Downloads and manages datasets for Telugu and Hindi SLM training"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.cache_dir = self.base_dir / "cache"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
              # Dataset URLs and configurations (Updated with working URLs)
        self.datasets = {
            # Using HuggingFace datasets API instead of direct URLs
            "cc100_telugu": {
                "url": "https://data.statmt.org/cc-100/te.txt.xz",
                "description": "CC-100 Telugu Corpus",
                "languages": ["te"],
                "type": "web_crawl",
                "size_mb": 400
            },
            "cc100_hindi": {
                "url": "https://data.statmt.org/cc-100/hi.txt.xz",
                "description": "CC-100 Hindi Corpus",
                "languages": ["hi"],
                "type": "web_crawl",
                "size_mb": 1500
            },
            # Alternative: Wikipedia dump-based datasets
            "wiki_te": {
                "hf_dataset": "wikimedia/wikipedia",
                "config": "20231101.te",
                "description": "Telugu Wikipedia Dump",
                "languages": ["te"],
                "type": "wikipedia",
                "size_mb": 50
            },
            "wiki_hi": {
                "hf_dataset": "wikimedia/wikipedia",
                "config": "20231101.hi",
                "description": "Hindi Wikipedia Dump",
                "languages": ["hi"],
                "type": "wikipedia",
                "size_mb": 200
            }
        }
          # Summarization datasets (Updated with working HuggingFace datasets)
        self.summarization_datasets = {
            "xlsum_telugu": {
                "hf_dataset": "csebuetnlp/xlsum",
                "config": "telugu",
                "description": "XLSum Telugu Summarization Dataset",
                "languages": ["te"],
                "type": "summarization"
            },
            "xlsum_hindi": {
                "hf_dataset": "csebuetnlp/xlsum",
                "config": "hindi",
                "description": "XLSum Hindi Summarization Dataset",
                "languages": ["hi"],
                "type": "summarization"
            },
            "indic_glue": {
                "hf_dataset": "ai4bharat/IndicGLUE",
                "config": "wnli.te",
                "description": "IndicGLUE Telugu NLI Dataset",
                "languages": ["te"],
                "type": "nli"
            }
        }

    def download_file(self, url: str, filename: str, max_retries: int = 3) -> Optional[Path]:
        """Download a file with retry logic and progress tracking"""
        file_path = self.raw_dir / filename
        
        # Skip if already downloaded
        if file_path.exists():
            logger.info(f"File {filename} already exists, skipping download")
            return file_path
            
        logger.info(f"Downloading {filename} from {url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progress indicator
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                
                print()  # New line after progress
                logger.info(f"Successfully downloaded {filename}")
                return file_path
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {filename}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait before retry
                else:
                    logger.error(f"Failed to download {filename} after {max_retries} attempts")
                    return None

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract zip/tar archives"""
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.gz', '.xz']:
                import gzip
                import lzma
                
                if archive_path.suffix.lower() == '.gz':
                    with gzip.open(archive_path, 'rt', encoding='utf-8') as f_in:
                        with open(extract_to / archive_path.stem, 'w', encoding='utf-8') as f_out:
                            f_out.write(f_in.read())
                else:  # .xz
                    with lzma.open(archive_path, 'rt', encoding='utf-8') as f_in:
                        with open(extract_to / archive_path.stem, 'w', encoding='utf-8') as f_out:
                            f_out.write(f_in.read())
            
            logger.info(f"Successfully extracted {archive_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path.name}: {e}")
            return False
    
    def download_huggingface_dataset(self, dataset_info: Dict) -> bool:
        """Download datasets from Hugging Face using datasets library"""
        try:
            from datasets import load_dataset
            
            dataset_name = dataset_info.get("hf_dataset")
            config = dataset_info.get("config")
            description = dataset_info.get("description", "")
            
            logger.info(f"Downloading {description} ({dataset_name})")
            
            # Load the dataset
            if config:
                dataset = load_dataset(dataset_name, config, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, trust_remote_code=True)
            
            # Create output directory
            safe_name = dataset_name.replace("/", "_") + (f"_{config}" if config else "")
            output_dir = self.raw_dir / safe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save all available splits
            for split_name, split_data in dataset.items():
                output_file = output_dir / f"{split_name}.jsonl"
                logger.info(f"Saving {len(split_data)} examples to {output_file}")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for example in split_data:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            logger.info(f"Successfully saved {dataset_name} to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False

    def download_all_datasets(self, include_large: bool = False) -> Dict[str, bool]:
        """Download all configured datasets"""
        results = {}
        
        logger.info("Starting dataset download process...")
        
        # Download text corpora
        for name, config in self.datasets.items():
            if not include_large and config.get("size_mb", 0) > 1000:
                logger.info(f"Skipping large dataset {name} (use --include-large to download)")
                results[name] = False
                continue
                
            logger.info(f"Processing {name}: {config['description']}")
            
            # Generate filename from URL
            parsed_url = urlparse(config["url"])
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = f"{name}.zip"
            
            # Download
            file_path = self.download_file(config["url"], filename)
            if file_path:
                # Extract if it's an archive
                if file_path.suffix.lower() in ['.zip', '.gz', '.xz']:
                    extract_path = self.raw_dir / name
                    results[name] = self.extract_archive(file_path, extract_path)
                else:
                    results[name] = True
            else:
                results[name] = False
        
        # Download summarization datasets
        for name, config in self.summarization_datasets.items():
            logger.info(f"Processing {name}: {config['description']}")
            
            parsed_url = urlparse(config["url"])
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = f"{name}.json"
            
            file_path = self.download_file(config["url"], filename)
            results[name] = file_path is not None
        
        # Try to download some HuggingFace datasets
        try:
            results["xlsum_hf"] = self.download_huggingface_dataset("xlsum", ["telugu", "hindi"])
        except:
            logger.warning("Could not download HuggingFace datasets (install datasets library)")
            results["xlsum_hf"] = False
        
        return results

    def get_dataset_info(self) -> Dict:
        """Get information about available datasets"""
        info = {
            "text_corpora": self.datasets,
            "summarization_datasets": self.summarization_datasets,
            "download_status": {}
        }
        
        # Check which files exist
        for name in self.datasets.keys():
            dataset_path = self.raw_dir / name
            info["download_status"][name] = dataset_path.exists()
            
        return info

    def create_manifest(self) -> Dict:
        """Create a manifest of all downloaded data"""
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": {},
            "statistics": {
                "total_files": 0,
                "total_size_mb": 0,
                "languages": set()
            }
        }
        
        # Scan raw directory
        for item in self.raw_dir.rglob("*"):
            if item.is_file():
                manifest["statistics"]["total_files"] += 1
                manifest["statistics"]["total_size_mb"] += item.stat().st_size / (1024 * 1024)
                
                # Try to determine language from filename
                for lang in ["te", "hi", "telugu", "hindi"]:
                    if lang in item.name.lower():
                        manifest["statistics"]["languages"].add(lang)
        
        manifest["statistics"]["languages"] = list(manifest["statistics"]["languages"])
        
        # Save manifest
        manifest_path = self.base_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created manifest at {manifest_path}")
        return manifest

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for AadiShakthiSLM")
    parser.add_argument("--data-dir", default="data", help="Base directory for datasets")
    parser.add_argument("--include-large", action="store_true", help="Include large datasets (>1GB)")
    parser.add_argument("--manifest-only", action="store_true", help="Only create manifest")
    parser.add_argument("--info", action="store_true", help="Show dataset information")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.info:
        info = downloader.get_dataset_info()
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return
    
    if args.manifest_only:
        manifest = downloader.create_manifest()
        print(f"Created manifest with {manifest['statistics']['total_files']} files")
        return
    
    # Download datasets
    results = downloader.download_all_datasets(include_large=args.include_large)
    
    # Show results
    print("\n" + "="*50)
    print("DOWNLOAD RESULTS")
    print("="*50)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {success_count}/{total_count} datasets downloaded successfully")
    
    # Create manifest
    manifest = downloader.create_manifest()
    print(f"\nManifest created with {manifest['statistics']['total_files']} files")
    print(f"Languages detected: {', '.join(manifest['statistics']['languages'])}")

if __name__ == "__main__":
    main()
