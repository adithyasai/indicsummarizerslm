#!/usr/bin/env py
"""
Production Environment Setup for AadiShakthiSLM
Ensures all dependencies and configurations are ready for training
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required. Found: {version.major}.{version.minor}")
        return False
    
    logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required packages"""
    requirements = [
        "torch>=1.11.0",
        "transformers>=4.20.0",
        "tokenizers>=0.13.0",
        "datasets>=2.0.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.1.0",
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "huggingface-hub>=0.10.0"
    ]
    
    logger.info("Installing production dependencies...")
    
    for package in requirements:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def verify_datasets():
    """Verify that training datasets are available"""
    logger.info("Verifying training datasets...")
    
    required_files = [
        "data/raw/hi.txt.xz",  # Hindi CC-100
        "data/raw/xlsum_hindi_combined.jsonl",
        "data/raw/xlsum_telugu_combined.jsonl", 
        "data/raw/samanantar_hindi_50k.jsonl",
        "data/raw/samanantar_telugu_50k.jsonl",
        "training_data/summarization/te_train.jsonl",
        "training_data/summarization/te_val.jsonl",
        "training_data/summarization/hi_train.jsonl",
        "training_data/summarization/hi_val.jsonl",
        "training_data/language_modeling/te_train.txt",
        "training_data/language_modeling/hi_train.txt"
    ]
    
    missing_files = []
    total_size = 0
    
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            logger.info(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            missing_files.append(file_path)
            logger.warning(f"❌ Missing: {file_path}")
    
    logger.info(f"Total dataset size: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    
    if missing_files:
        logger.error(f"Missing {len(missing_files)} required dataset files")
        return False
    
    logger.info("✅ All required datasets available")
    return True

def verify_tokenizer():
    """Verify custom tokenizer is trained and ready"""
    logger.info("Verifying custom IndicTokenizer...")
    
    tokenizer_files = [
        "models/indic_tokenizer/vocab.json",
        "models/indic_tokenizer/merges.txt", 
        "models/indic_tokenizer/tokenizer_config.json"
    ]
    
    for file_path in tokenizer_files:
        if not Path(file_path).exists():
            logger.error(f"❌ Missing tokenizer file: {file_path}")
            return False
        logger.info(f"✅ {file_path}")
    
    # Test tokenizer functionality
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.indic_tokenizer import IndicTokenizer
        
        tokenizer = IndicTokenizer.from_pretrained("models/indic_tokenizer")
        
        # Test Telugu
        te_text = "తెలుగు భాష చాలా అందమైనది"
        te_encoded = tokenizer.encode(te_text)
        te_decoded = tokenizer.decode(te_encoded.ids)
        
        # Test Hindi  
        hi_text = "हिंदी भाषा बहुत सुंदर है"
        hi_encoded = tokenizer.encode(hi_text)
        hi_decoded = tokenizer.decode(hi_encoded.ids)
        
        logger.info(f"✅ Tokenizer test passed")
        logger.info(f"   Telugu: '{te_text}' → {len(te_encoded.ids)} tokens → '{te_decoded}'")
        logger.info(f"   Hindi: '{hi_text}' → {len(hi_encoded.ids)} tokens → '{hi_decoded}'")
        logger.info(f"   Vocabulary size: {tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tokenizer test failed: {e}")
        return False

def verify_model_architecture():
    """Verify SLM model can be instantiated"""
    logger.info("Verifying SLM model architecture...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from models.slm_model import SLMModel, SLMConfig
        
        # Test configuration
        config = SLMConfig(
            vocab_size=32000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12
        )
        
        # Test model instantiation
        model = SLMModel(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"✅ SLM model verified")
        logger.info(f"   Architecture: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        logger.info(f"   Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        logger.info(f"   Vocab size: {config.vocab_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability and memory"""
    logger.info("Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test GPU functionality
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor.t())
            logger.info(f"✅ GPU computation test passed")
            
            return True
        else:
            logger.warning("⚠️  No GPU available - training will use CPU (slower)")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU check failed: {e}")
        return False

def create_training_directories():
    """Create necessary directories for training"""
    logger.info("Creating training directories...")
    
    directories = [
        "models/checkpoints",
        "models/final_models",
        "logs",
        "results"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ {dir_path}")

def generate_training_commands():
    """Generate optimized training commands"""
    logger.info("Generating training commands...")
    
    commands = {
        "Telugu Summarization": "python train_production_slm.py --task summarization --language te --config production_config.json --epochs 3",
        "Hindi Summarization": "python train_production_slm.py --task summarization --language hi --config production_config.json --epochs 3",
        "Telugu Language Modeling": "python train_production_slm.py --task language_modeling --language te --config production_config.json --epochs 2",
        "Hindi Language Modeling": "python train_production_slm.py --task language_modeling --language hi --config production_config.json --epochs 2"
    }
    
    commands_file = Path("training_commands.txt")
    with open(commands_file, 'w', encoding='utf-8') as f:
        f.write("# AadiShakthiSLM Production Training Commands\\n")
        f.write("# Execute these commands in order for full training pipeline\\n\\n")
        
        for i, (name, command) in enumerate(commands.items(), 1):
            f.write(f"# Step {i}: {name}\\n")
            f.write(f"{command}\\n\\n")
    
    logger.info(f"✅ Training commands saved to {commands_file}")

def main():
    """Main setup verification"""
    logger.info("🚀 AadiShakthiSLM Production Setup Verification")
    logger.info("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Verify Datasets", verify_datasets),
        ("Verify Tokenizer", verify_tokenizer),
        ("Verify Model", verify_model_architecture),
        ("Check GPU", check_gpu_availability),
        ("Create Directories", create_training_directories),
        ("Generate Commands", generate_training_commands)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        logger.info(f"\\n{'='*40}")
        logger.info(f"Checking: {name}")
        logger.info(f"{'='*40}")
        
        try:
            if check_func():
                passed += 1
                logger.info(f"✅ {name}: PASSED")
            else:
                logger.error(f"❌ {name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {name}: ERROR - {e}")
    
    # Final summary
    logger.info(f"\\n{'='*60}")
    logger.info("SETUP VERIFICATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Passed: {passed}/{total} checks")
    
    if passed == total:
        logger.info("🎉 ALL CHECKS PASSED - READY FOR PRODUCTION TRAINING!")
        logger.info("\\n📋 Next Steps:")
        logger.info("1. Review training_commands.txt")
        logger.info("2. Start with: python train_production_slm.py --task summarization --language te --epochs 3")
        logger.info("3. Monitor training progress in logs/")
        logger.info("4. Models will be saved to models/")
        return True
    else:
        logger.error(f"❌ {total - passed} checks failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
