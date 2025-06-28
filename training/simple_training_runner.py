#!/usr/bin/env python3
"""
Simple Training Runner for AadiShakti SLM
Runs training with basic logging to avoid Unicode issues
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path

# Configure logging without fancy Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure UTF-8 encoding for console output
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def run_data_preparation():
    """Run data preparation"""
    logger.info("=== STEP 1: DATA PREPARATION ===")
    
    try:
        result = subprocess.run([
            sys.executable, "prepare_training_data.py"
        ], capture_output=True, text=True, encoding='utf-8', timeout=600)
        
        if result.returncode == 0:
            logger.info("SUCCESS: Data preparation completed")
            return True
        else:
            logger.error(f"FAILED: Data preparation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"ERROR: Data preparation error: {e}")
        return False

def check_tokenizer():
    """Check if tokenizer exists"""
    tokenizer_files = [
        "models/indic_tokenizer/vocab.json",
        "models/indic_tokenizer/merges.txt",
        "models/indic_tokenizer/tokenizer_config.json"
    ]
    
    return all(os.path.exists(f) for f in tokenizer_files)

def run_simple_training():
    """Run simple incremental training"""
    logger.info("=== STEP 2: MODEL TRAINING ===")
    
    if not check_tokenizer():
        logger.error("FAILED: Tokenizer not found. Please run tokenizer training first.")
        return False
    
    try:
        logger.info("Starting incremental training process...")
        logger.info("This will run in small chunks to avoid system overload")
        
        # Run the incremental training
        process = subprocess.Popen([
            sys.executable, "incremental_training.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
           text=True, encoding='utf-8')
        
        # Monitor output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Clean output of Unicode characters that might cause issues
                clean_output = output.encode('ascii', 'ignore').decode('ascii').strip()
                if clean_output:
                    logger.info(f"[TRAINING] {clean_output}")
        
        return_code = process.poll()
        
        if return_code == 0:
            logger.info("SUCCESS: Training completed successfully")
            return True
        else:
            logger.error(f"FAILED: Training failed with return code: {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"ERROR: Training error: {e}")
        return False

def check_training_results():
    """Check if training produced results"""
    checkpoint_dir = "models/checkpoints/latest"
    
    required_files = [
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
        os.path.join(checkpoint_dir, "config.json")
    ]
    
    if all(os.path.exists(f) for f in required_files):
        logger.info("SUCCESS: Training files found")
        
        # Check training state
        state_file = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                steps = state.get('step', 0)
                logger.info(f"Training completed with {steps} steps")
                return True
            except Exception as e:
                logger.warning(f"Could not read training state: {e}")
        
        return True
    else:
        logger.warning("WARNING: Some training files missing")
        return False

def main():
    """Main training function"""
    logger.info("STARTING AADISHAKTI SLM TRAINING")
    logger.info("=" * 50)
    
    success = True
    
    # Step 1: Data preparation
    if not run_data_preparation():
        logger.error("Data preparation failed - continuing with existing data")
        # Don't fail completely, might have existing data
    
    # Step 2: Training
    if not run_simple_training():
        logger.error("Training failed")
        success = False
    
    # Step 3: Check results
    if not check_training_results():
        logger.warning("Training results incomplete")
    
    # Summary
    logger.info("=" * 50)
    if success:
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("Model saved to: models/checkpoints/latest/")
        logger.info("Next steps:")
        logger.info("1. Test the model with the API")
        logger.info("2. Evaluate summarization quality")
        logger.info("3. Run additional training if needed")
    else:
        logger.error("TRAINING FAILED!")
        logger.error("Check the logs above for error details")
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
