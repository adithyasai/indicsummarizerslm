#!/usr/bin/env py
"""
Safe Lightweight Training for AadiShakthiSLM
Designed to avoid freezing the laptop with careful resource management
"""

import os
import sys
import json
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafeTrainingManager:
    """Safe training manager with resource monitoring"""
    
    def __init__(self):
        self.max_memory_percent = 70  # Don't use more than 70% RAM
        self.max_cpu_percent = 80     # Don't use more than 80% CPU
        self.training_active = False
        self.emergency_stop = False
        
    def check_system_resources(self) -> Dict:
        """Check current system resource usage"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "cpu_percent": cpu_percent,
            "safe_to_continue": (
                memory.percent < self.max_memory_percent and 
                cpu_percent < self.max_cpu_percent
            )
        }
    
    def emergency_cleanup(self):
        """Emergency cleanup to free resources"""
        logger.warning("🚨 Emergency cleanup triggered!")
        gc.collect()  # Force garbage collection
        time.sleep(2)  # Give system time to recover
        
    def safe_training_step(self, step_function, *args, **kwargs):
        """Execute a training step with safety checks"""
        # Check resources before step
        resources = self.check_system_resources()
        
        if not resources["safe_to_continue"]:
            logger.warning(f"⚠️ High resource usage detected:")
            logger.warning(f"   Memory: {resources['memory_percent']:.1f}%")
            logger.warning(f"   CPU: {resources['cpu_percent']:.1f}%")
            logger.warning("   Pausing for system recovery...")
            
            self.emergency_cleanup()
            time.sleep(5)  # Wait for system to recover
            
            # Recheck
            resources = self.check_system_resources()
            if not resources["safe_to_continue"]:
                logger.error("❌ System resources still high. Stopping training.")
                self.emergency_stop = True
                return None
        
        try:
            result = step_function(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"❌ Training step failed: {e}")
            self.emergency_cleanup()
            return None

class LightweightSLMTrainer:
    """Lightweight trainer that won't freeze the system"""
    
    def __init__(self):
        self.safety_manager = SafeTrainingManager()
        self.device = "cpu"  # Force CPU to avoid GPU memory issues
        self.tokenizer = None
        
        # Very conservative settings
        self.config = {
            "batch_size": 1,           # Smallest possible batch
            "max_length": 256,         # Shorter sequences
            "learning_rate": 1e-4,     # Lower learning rate
            "num_epochs": 1,           # Just 1 epoch to start
            "save_every": 50,          # Save frequently
            "log_every": 10,           # Log frequently
            "cleanup_every": 25        # Cleanup memory frequently
        }
        
        logger.info("🛡️ Lightweight trainer initialized with safety features")
    
    def load_tokenizer(self):
        """Load custom tokenizer safely"""
        try:
            sys.path.insert(0, str(Path.cwd()))
            from src.indic_tokenizer import IndicTokenizer
            
            tokenizer_path = Path("models/indic_tokenizer")
            if tokenizer_path.exists():
                self.tokenizer = IndicTokenizer.from_pretrained(str(tokenizer_path))
                logger.info(f"✅ Loaded custom tokenizer: {self.tokenizer.vocab_size} tokens")
                return True
            else:
                logger.error("❌ Custom tokenizer not found")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            return False
    
    def load_small_dataset(self, language: str = "te", max_samples: int = 100):
        """Load a very small subset of data for safe training"""
        try:
            data_file = Path(f"training_data/summarization/{language}_train.jsonl")
            
            if not data_file.exists():
                logger.error(f"❌ Training data not found: {data_file}")
                return []
            
            samples = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                        
                    try:
                        item = json.loads(line)
                        if "input" in item and "target" in item:
                            samples.append({
                                "input_text": item["input"][:500],  # Truncate for safety
                                "target_text": item["target"][:200],
                                "language": language
                            })
                    except:
                        continue
            
            logger.info(f"✅ Loaded {len(samples)} safe training samples")
            return samples
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return []
    
    def create_simple_model(self):
        """Create a very simple model for safe training"""
        try:
            # Use basic torch operations to avoid complex dependencies
            logger.info("Creating simple sequence-to-sequence model...")
            
            # For now, we'll create a simple mapping approach
            # that learns basic patterns without heavy computation
            class SimplePatternLearner:
                def __init__(self):
                    self.patterns = {}
                    self.language_patterns = {"te": {}, "hi": {}}
                
                def learn_pattern(self, input_text, target_text, language):
                    """Learn simple extraction patterns"""
                    # Simple heuristic learning
                    input_words = input_text.split()
                    target_words = target_text.split()
                    
                    # Store common patterns
                    if len(input_words) > 10 and len(target_words) > 3:
                        ratio = len(target_words) / len(input_words)
                        
                        if language not in self.language_patterns:
                            self.language_patterns[language] = {"ratios": [], "key_words": set()}
                        
                        self.language_patterns[language]["ratios"].append(ratio)
                        self.language_patterns[language]["key_words"].update(target_words[:5])
                
                def generate_summary(self, text, language="te", max_length=150):
                    """Generate summary using learned patterns"""
                    sentences = text.split('.')
                    if not sentences:
                        return text[:max_length]
                    
                    # Use learned patterns to select best sentences
                    if language in self.language_patterns:
                        patterns = self.language_patterns[language]
                        key_words = patterns.get("key_words", set())
                        
                        # Score sentences based on key words
                        scored_sentences = []
                        for sentence in sentences:
                            score = sum(1 for word in sentence.split() if word in key_words)
                            scored_sentences.append((score, sentence))
                        
                        # Select top sentences
                        scored_sentences.sort(reverse=True)
                        summary_sentences = [s[1] for s in scored_sentences[:3]]
                        summary = '. '.join(summary_sentences).strip()
                        
                        if len(summary) > max_length:
                            summary = summary[:max_length] + "..."
                        
                        return summary
                    
                    # Fallback to first few sentences
                    return '. '.join(sentences[:2])[:max_length]
            
            self.model = SimplePatternLearner()
            logger.info("✅ Simple pattern learning model created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create model: {e}")
            return False
    
    def safe_train(self, language: str = "te"):
        """Safe training process with continuous monitoring"""
        logger.info(f"🚀 Starting safe training for {language}")
        logger.info("🛡️ Safety features enabled:")
        logger.info(f"   Max RAM usage: {self.safety_manager.max_memory_percent}%")
        logger.info(f"   Max CPU usage: {self.safety_manager.max_cpu_percent}%")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch size: {self.config['batch_size']}")
        
        # Step 1: Load tokenizer
        if not self.load_tokenizer():
            return False
        
        # Step 2: Load small dataset
        samples = self.load_small_dataset(language, max_samples=100)
        if not samples:
            return False
        
        # Step 3: Create simple model
        if not self.create_simple_model():
            return False
        
        # Step 4: Safe training loop
        logger.info("📚 Starting safe training loop...")
        
        trained_samples = 0
        start_time = time.time()
        
        for epoch in range(self.config["num_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            for i, sample in enumerate(samples):
                # Safety check before each sample
                if self.safety_manager.emergency_stop:
                    logger.error("🚨 Emergency stop triggered!")
                    break
                
                # Train on this sample safely
                def train_step():
                    self.model.learn_pattern(
                        sample["input_text"],
                        sample["target_text"],
                        sample["language"]
                    )
                    return True
                
                result = self.safety_manager.safe_training_step(train_step)
                
                if result is None:
                    logger.error("❌ Training step failed, stopping")
                    break
                
                trained_samples += 1
                
                # Periodic logging and cleanup
                if (i + 1) % self.config["log_every"] == 0:
                    elapsed = time.time() - start_time
                    resources = self.safety_manager.check_system_resources()
                    
                    logger.info(f"Step {i + 1}/{len(samples)}: "
                              f"RAM {resources['memory_percent']:.1f}%, "
                              f"CPU {resources['cpu_percent']:.1f}%, "
                              f"Time: {elapsed:.1f}s")
                
                if (i + 1) % self.config["cleanup_every"] == 0:
                    gc.collect()
                    time.sleep(0.5)  # Brief pause
        
        # Step 5: Save the trained model
        self.save_simple_model(language)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Safe training completed!")
        logger.info(f"   Samples trained: {trained_samples}")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   Language: {language}")
        
        return True
    
    def save_simple_model(self, language: str):
        """Save the simple trained model"""
        try:
            save_dir = Path("models") / f"safe_model_{language}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model patterns
            model_data = {
                "language": language,
                "patterns": self.model.language_patterns,
                "config": self.config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(save_dir / "model.json", 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Model saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")

    def test_trained_model(self, language: str, test_text: str):
        """Test the trained model safely"""
        try:
            if hasattr(self, 'model') and self.model:
                summary = self.model.generate_summary(test_text, language, max_length=150)
                
                logger.info("🧪 Test Results:")
                logger.info(f"Input ({len(test_text)} chars): {test_text[:100]}...")
                logger.info(f"Output ({len(summary)} chars): {summary}")
                
                return summary
            else:
                logger.error("❌ No trained model available")
                return test_text[:150]
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return test_text[:150]

def main():
    """Main safe training function"""
    logger.info("🛡️ AadiShakthiSLM - Safe Training Mode")
    logger.info("=" * 50)
    logger.info("Designed to avoid system freezing")
    
    # Check initial system state
    trainer = LightweightSLMTrainer()
    resources = trainer.safety_manager.check_system_resources()
    
    logger.info(f"Initial system state:")
    logger.info(f"   RAM: {resources['memory_percent']:.1f}% used")
    logger.info(f"   CPU: {resources['cpu_percent']:.1f}% used")
    logger.info(f"   Available RAM: {resources['memory_available_gb']:.1f} GB")
    
    if not resources["safe_to_continue"]:
        logger.error("❌ System resources too high to start training safely")
        logger.error("Please close other applications and try again")
        return False
    
    # Start safe training
    try:
        # Train Telugu first (smaller dataset)
        logger.info("\n🚀 Starting Telugu training...")
        success_te = trainer.safe_train("te")
        
        if success_te:
            logger.info("✅ Telugu training completed successfully!")
            
            # Test with user's text
            test_text = """నేటి తెలంగాణ రాష్ట్రంలో వాతావరణ పరిస్థితులు అత్యంత తీవ్రంగా మారుతున్నాయి. నైరుతి రుతుపవనాల ప్రభావంతో రాష్ట్రవ్యాప్తంగా భారీ వర్షాలు కురుస్తున్నాయి. హైదరాబాద్‌, వరంగల్‌, ఖమ్మం వంటి జిల్లాల్లో ఎల్లో అలర్ట్ ప్రకటించబడింది."""
            
            logger.info("\n🧪 Testing with your text...")
            improved_summary = trainer.test_trained_model("te", test_text)
            
            logger.info(f"\n🎉 Training and testing completed!")
            logger.info(f"The model should now provide better summaries.")
            
        else:
            logger.error("❌ Telugu training failed")
        
        return success_te
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        trainer.safety_manager.emergency_cleanup()
        return False
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        trainer.safety_manager.emergency_cleanup()
        return False

if __name__ == "__main__":
    main()
