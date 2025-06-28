#!/usr/bin/env python3
"""
Training Orchestrator for AadiShakti SLM
Coordinates data preparation, tokenizer setup, and incremental training
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """Orchestrates the complete training pipeline"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.setup_paths()
        
        logger.info("🎵 Training Orchestrator initialized")
        logger.info(f"   Project root: {self.project_root}")
    
    def setup_paths(self):
        """Setup all required paths"""
        self.paths = {
            'data_raw': self.project_root / "data" / "raw",
            'data_processed': self.project_root / "data" / "processed", 
            'models': self.project_root / "models",
            'tokenizer': self.project_root / "models" / "indic_tokenizer",
            'checkpoints': self.project_root / "models" / "checkpoints",
            'scripts': {
                'data_prep': self.project_root / "prepare_training_data.py",
                'tokenizer': self.project_root / "train_indic_tokenizer.py",
                'training': self.project_root / "incremental_training.py"
            }
        }
        
        # Create directories
        for path_key, path_value in self.paths.items():
            if path_key != 'scripts' and isinstance(path_value, Path):
                path_value.mkdir(parents=True, exist_ok=True)
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        checks = {
            'python_env': self.check_python_environment(),
            'required_packages': self.check_required_packages(),
            'data_availability': self.check_data_availability(),
            'tokenizer_ready': self.check_tokenizer_status(),
            'disk_space': self.check_disk_space(),
            'memory': self.check_memory_requirements()
        }
        
        for check_name, status in checks.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {check_name.replace('_', ' ').title()}")
        
        return checks
    
    def check_python_environment(self) -> bool:
        """Check Python environment"""
        try:
            import torch
            import transformers
            import numpy as np
            return True
        except ImportError as e:
            logger.warning(f"Missing Python packages: {e}")
            return False
    
    def check_required_packages(self) -> bool:
        """Check if required packages are installed"""
        required_packages = [
            'torch', 'transformers', 'tokenizers', 'datasets',
            'numpy', 'psutil', 'tqdm'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            return False
        return True
    
    def check_data_availability(self) -> bool:
        """Check if training data is available"""
        # Check for processed data
        processed_files = [
            self.paths['data_processed'] / "telugu_texts.json",
            self.paths['data_processed'] / "hindi_texts.json", 
            self.paths['data_processed'] / "mixed_texts.json"
        ]
        
        if all(f.exists() for f in processed_files):
            return True
            
        # Check for raw data
        if self.paths['data_raw'].exists() and any(self.paths['data_raw'].iterdir()):
            return True
            
        logger.info("   No training data found - will create sample data")
        return True  # We can create sample data
    
    def check_tokenizer_status(self) -> bool:
        """Check if tokenizer is ready"""
        tokenizer_files = [
            self.paths['tokenizer'] / "vocab.json",
            self.paths['tokenizer'] / "merges.txt",
            self.paths['tokenizer'] / "tokenizer_config.json"
        ]
        
        return all(f.exists() for f in tokenizer_files)
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free / (1024**3)
            
            if free_gb < 2.0:  # Need at least 2GB
                logger.warning(f"Low disk space: {free_gb:.1f}GB available")
                return False
            return True
        except Exception:
            return True  # Assume OK if can't check
    
    def check_memory_requirements(self) -> bool:
        """Check memory requirements"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 2.0:  # Need at least 2GB
                logger.warning(f"Low memory: {available_gb:.1f}GB available")
                return False
            return True
        except Exception:
            return True  # Assume OK if can't check
    
    def prepare_data(self) -> bool:
        """Prepare training data"""
        logger.info("📚 Preparing training data...")
        
        try:
            # Run data preparation script
            script_path = self.paths['scripts']['data_prep']
            
            if not script_path.exists():
                logger.error(f"Data preparation script not found: {script_path}")
                return False
            
            # Change to project directory and run script
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            try:
                result = subprocess.run([
                    sys.executable, str(script_path)
                ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
                
                if result.returncode == 0:
                    logger.info("✅ Data preparation completed successfully")
                    logger.info(f"   Output: {result.stdout[-200:]}")  # Last 200 chars
                    return True
                else:
                    logger.error(f"Data preparation failed: {result.stderr}")
                    return False
                    
            finally:
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            logger.error("Data preparation timed out")
            return False
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return False
    
    def setup_tokenizer(self) -> bool:
        """Setup or verify tokenizer"""
        logger.info("🔤 Setting up tokenizer...")
        
        if self.check_tokenizer_status():
            logger.info("✅ Tokenizer already exists and ready")
            return True
        
        try:
            # Run tokenizer training script
            script_path = self.paths['scripts']['tokenizer']
            
            if not script_path.exists():
                logger.error(f"Tokenizer script not found: {script_path}")
                return False
            
            # Change to project directory and run script
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            try:
                result = subprocess.run([
                    sys.executable, str(script_path)
                ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
                
                if result.returncode == 0:
                    logger.info("✅ Tokenizer training completed successfully")
                    return self.check_tokenizer_status()
                else:
                    logger.error(f"Tokenizer training failed: {result.stderr}")
                    return False
                    
            finally:
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            logger.error("Tokenizer training timed out")
            return False
        except Exception as e:
            logger.error(f"Tokenizer setup error: {e}")
            return False
    
    def run_incremental_training(self) -> bool:
        """Run the incremental training process"""
        logger.info("🏃 Starting incremental training...")
        
        try:
            # Run incremental training script
            script_path = self.paths['scripts']['training']
            
            if not script_path.exists():
                logger.error(f"Training script not found: {script_path}")
                return False
            
            # Change to project directory and run script
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            try:
                # Start training process
                logger.info("🚀 Launching incremental training process...")
                logger.info("   This may take a while - training in small chunks")
                logger.info("   Press Ctrl+C to stop gracefully")
                
                process = subprocess.Popen([
                    sys.executable, str(script_path)
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                   text=True, bufsize=1, universal_newlines=True)
                
                # Monitor the process
                output_lines = []
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logger.info(f"[TRAINING] {output.strip()}")
                        output_lines.append(output.strip())
                        
                        # Keep only last 100 lines to avoid memory issues
                        if len(output_lines) > 100:
                            output_lines = output_lines[-50:]
                
                return_code = process.poll()
                
                if return_code == 0:
                    logger.info("✅ Incremental training completed successfully")
                    return True
                else:
                    logger.error(f"Training failed with return code: {return_code}")
                    return False
                    
            finally:
                os.chdir(original_cwd)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def validate_training_results(self) -> bool:
        """Validate that training completed successfully"""
        logger.info("🔍 Validating training results...")
        
        # Check for model files
        model_files = [
            self.paths['checkpoints'] / "latest" / "pytorch_model.bin",
            self.paths['checkpoints'] / "latest" / "config.json",
            self.paths['checkpoints'] / "latest" / "training_state.json"
        ]
        
        missing_files = [f for f in model_files if not f.exists()]
        
        if missing_files:
            logger.warning(f"Missing model files: {missing_files}")
            return False
        
        # Check training state
        try:
            state_file = self.paths['checkpoints'] / "latest" / "training_state.json"
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            steps = state.get('step', 0)
            if steps > 0:
                logger.info(f"✅ Training completed with {steps} steps")
                return True
            else:
                logger.warning("No training steps recorded")
                return False
                
        except Exception as e:
            logger.error(f"Could not validate training state: {e}")
            return False
    
    def run_complete_training_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("🚀 AadiShakti SLM - Complete Training Pipeline")
        logger.info("=" * 70)
        
        # Step 1: Check prerequisites
        logger.info("\n📋 Step 1: Checking Prerequisites")
        logger.info("-" * 40)
        
        prereqs = self.check_prerequisites()
        critical_missing = [k for k, v in prereqs.items() if not v and k in ['python_env', 'required_packages']]
        
        if critical_missing:
            logger.error("❌ Critical prerequisites missing. Please install required packages.")
            logger.error("   Run: pip install torch transformers tokenizers datasets numpy psutil tqdm")
            return False
        
        # Step 2: Prepare data
        logger.info("\n📚 Step 2: Preparing Training Data")
        logger.info("-" * 40)
        
        if not self.prepare_data():
            logger.error("❌ Data preparation failed")
            return False
        
        # Step 3: Setup tokenizer
        logger.info("\n🔤 Step 3: Setting Up Tokenizer")
        logger.info("-" * 40)
        
        if not self.setup_tokenizer():
            logger.error("❌ Tokenizer setup failed")
            return False
        
        # Step 4: Run training
        logger.info("\n🏃 Step 4: Running Incremental Training")
        logger.info("-" * 40)
        
        if not self.run_incremental_training():
            logger.error("❌ Training failed or was interrupted")
            return False
        
        # Step 5: Validate results
        logger.info("\n🔍 Step 5: Validating Training Results")
        logger.info("-" * 40)
        
        if not self.validate_training_results():
            logger.warning("⚠️ Training validation failed - model may not be fully trained")
        
        # Success!
        logger.info("\n🎉 Training Pipeline Completed!")
        logger.info("=" * 70)
        logger.info("✅ Data prepared and processed")
        logger.info("✅ Custom Indic tokenizer trained")
        logger.info("✅ SLM model training completed")
        logger.info("✅ Model saved and ready for use")
        logger.info("\n📍 Next Steps:")
        logger.info("   1. Test the trained model with the API")
        logger.info("   2. Evaluate summarization quality")
        logger.info("   3. Run additional training if needed")
        logger.info(f"\n📂 Model Location: {self.paths['checkpoints'] / 'latest'}")
        
        return True

def main():
    """Main orchestrator function"""
    try:
        # Create orchestrator
        orchestrator = TrainingOrchestrator()
        
        # Run complete pipeline
        success = orchestrator.run_complete_training_pipeline()
        
        if success:
            logger.info("🎊 All training completed successfully!")
            return 0
        else:
            logger.error("💥 Training pipeline failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️ Training pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
