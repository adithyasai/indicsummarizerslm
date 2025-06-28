#!/usr/bin/env py
"""
Master Data Pipeline for AadiShakthiSLM
Orchestrates the complete data collection, preprocessing, and training data creation process
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import logging
import argparse
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterDataPipeline:
    """Orchestrates the complete data pipeline for AadiShakthiSLM"""
    
    def __init__(self, base_dir: str = ".", data_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.data_dir = Path(data_dir)
        self.scripts_dir = self.base_dir / "data_collection"
        
        # Ensure scripts directory exists
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline stages
        self.stages = {
            "download": {
                "script": "download_datasets.py",
                "description": "Download Telugu and Hindi datasets",
                "required_packages": ["requests", "datasets"]
            },
            "preprocess": {
                "script": "preprocess_data.py", 
                "description": "Clean and preprocess text data",
                "required_packages": ["pandas", "scikit-learn"]
            },
            "create_training": {
                "script": "create_training_data.py",
                "description": "Create training/validation/test splits",
                "required_packages": ["numpy", "pandas", "scikit-learn"]
            }
        }
        
        # Track pipeline state
        self.state_file = self.data_dir / "pipeline_state.json"
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load pipeline state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        
        return {
            "completed_stages": [],
            "last_run": None,
            "errors": [],
            "statistics": {}
        }

    def save_state(self) -> None:
        """Save pipeline state to file"""
        self.state["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save state file: {e}")

    def check_dependencies(self, packages: List[str]) -> List[str]:
        """Check if required packages are installed"""
        missing = []
        
        for package in packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        
        return missing

    def install_packages(self, packages: List[str]) -> bool:
        """Install required packages"""
        if not packages:
            return True
        
        logger.info(f"Installing packages: {', '.join(packages)}")
        
        try:
            # Try different Python commands
            python_cmd = None
            for cmd in [sys.executable, "py -3.11", "py", "python"]:  # Removed python3
                try:
                    subprocess.check_call([cmd.split()[0] if " " in cmd else cmd, "--version"], 
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    python_cmd = cmd
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if not python_cmd:
                logger.error("No Python interpreter found")
                return False
            
            # Install packages
            cmd_parts = python_cmd.split() + ["-m", "pip", "install"] + packages
            result = subprocess.run(cmd_parts, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Packages installed successfully")
                return True
            else:
                logger.error(f"Failed to install packages: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing packages: {e}")
            return False

    def run_stage(self, stage_name: str, **kwargs) -> bool:
        """Run a specific pipeline stage"""
        if stage_name not in self.stages:
            logger.error(f"Unknown stage: {stage_name}")
            return False
        
        stage = self.stages[stage_name]
        script_path = self.scripts_dir / stage["script"]
        
        logger.info(f"Running stage: {stage_name}")
        logger.info(f"Description: {stage['description']}")
        
        # Check if script exists
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        # Check and install dependencies
        missing_packages = self.check_dependencies(stage["required_packages"])
        if missing_packages:
            logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
            if not self.install_packages(missing_packages):
                logger.error(f"Failed to install dependencies for {stage_name}")
                return False
        
        # Build command
        python_cmd = sys.executable
        cmd = [python_cmd, str(script_path)]
        
        # Add common arguments
        if stage_name == "download":
            cmd.extend(["--data-dir", str(self.data_dir)])
            if kwargs.get("include_large", False):
                cmd.append("--include-large")
        elif stage_name == "preprocess":
            cmd.extend(["--data-dir", str(self.data_dir)])
            if kwargs.get("workers"):
                cmd.extend(["--workers", str(kwargs["workers"])])
        elif stage_name == "create_training":
            cmd.extend(["--data-dir", str(self.data_dir)])
            cmd.extend(["--output-dir", str(self.data_dir / "training_data")])
            if kwargs.get("max_samples"):
                cmd.extend(["--max-samples", str(kwargs["max_samples"])])
        
        # Run the script
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            end_time = time.time()
            
            # Log output
            if result.stdout:
                logger.info(f"Output from {stage_name}:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            
            if result.stderr:
                logger.warning(f"Errors from {stage_name}:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.warning(f"  {line}")
            
            if result.returncode == 0:
                logger.info(f"Stage {stage_name} completed successfully in {end_time - start_time:.1f}s")
                
                # Update state
                if stage_name not in self.state["completed_stages"]:
                    self.state["completed_stages"].append(stage_name)
                self.state["statistics"][stage_name] = {
                    "duration": end_time - start_time,
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.save_state()
                return True
            else:
                logger.error(f"Stage {stage_name} failed with return code {result.returncode}")
                self.state["errors"].append({
                    "stage": stage_name,
                    "error": result.stderr,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                self.save_state()
                return False
                
        except Exception as e:
            logger.error(f"Error running {stage_name}: {e}")
            self.state["errors"].append({
                "stage": stage_name,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            self.save_state()
            return False

    def run_full_pipeline(self, **kwargs) -> bool:
        """Run the complete data pipeline"""
        logger.info("Starting complete data pipeline for AadiShakthiSLM")
        
        # Print current configuration
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Include large datasets: {kwargs.get('include_large', False)}")
        
        total_start = time.time()
        successful_stages = 0
        
        # Run each stage in order
        for stage_name in ["download", "preprocess", "create_training"]:
            if kwargs.get("skip_completed", True) and stage_name in self.state["completed_stages"]:
                logger.info(f"Skipping {stage_name} (already completed)")
                successful_stages += 1
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE {successful_stages + 1}/3: {stage_name.upper()}")
            logger.info(f"{'='*60}")
            
            if self.run_stage(stage_name, **kwargs):
                successful_stages += 1
            else:
                logger.error(f"Pipeline failed at stage: {stage_name}")
                if not kwargs.get("continue_on_error", False):
                    break
        
        total_time = time.time() - total_start
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Completed stages: {successful_stages}/3")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        if successful_stages == 3:
            logger.info("✅ Pipeline completed successfully!")
            
            # Show data statistics
            self.show_data_statistics()
            
            return True
        else:
            logger.error("❌ Pipeline failed or incomplete")
            return False

    def show_data_statistics(self) -> None:
        """Show statistics about the collected data"""
        try:
            # Check data directories
            raw_dir = self.data_dir / "raw"
            processed_dir = self.data_dir / "processed"
            training_dir = self.data_dir / "training_data"
            
            logger.info("\nDATA STATISTICS:")
            logger.info("-" * 40)
            
            if raw_dir.exists():
                raw_files = list(raw_dir.rglob("*"))
                raw_size = sum(f.stat().st_size for f in raw_files if f.is_file()) / 1024 / 1024
                logger.info(f"Raw data: {len(raw_files)} files, {raw_size:.1f} MB")
            
            if processed_dir.exists():
                processed_files = list(processed_dir.glob("*.jsonl"))
                processed_size = sum(f.stat().st_size for f in processed_files) / 1024 / 1024
                logger.info(f"Processed data: {len(processed_files)} files, {processed_size:.1f} MB")
            
            if training_dir.exists():
                training_files = list(training_dir.rglob("*.jsonl")) + list(training_dir.rglob("*.txt"))
                training_size = sum(f.stat().st_size for f in training_files) / 1024 / 1024
                logger.info(f"Training data: {len(training_files)} files, {training_size:.1f} MB")
            
            # Check for specific result files
            manifest_file = self.data_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                logger.info(f"Languages detected: {', '.join(manifest['statistics'].get('languages', []))}")
            
            pipeline_results = training_dir / "pipeline_results.json"
            if pipeline_results.exists():
                with open(pipeline_results, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                stats = results.get("statistics", {})
                logger.info(f"Language modeling samples: {stats.get('total_lm_samples', 0):,}")
                logger.info(f"Summarization samples: {stats.get('total_sum_samples', 0):,}")
            
        except Exception as e:
            logger.warning(f"Could not generate statistics: {e}")

    def reset_pipeline(self) -> None:
        """Reset pipeline state"""
        logger.info("Resetting pipeline state...")
        self.state = {
            "completed_stages": [],
            "last_run": None,
            "errors": [],
            "statistics": {}
        }
        self.save_state()
        logger.info("Pipeline state reset")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AadiShakthiSLM Data Pipeline")
    
    # Basic options
    parser.add_argument("--data-dir", default="data", help="Base directory for datasets")
    parser.add_argument("--include-large", action="store_true", help="Include large datasets (>1GB)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for preprocessing")
    parser.add_argument("--max-samples", type=int, default=100000, help="Maximum samples per language")
    
    # Pipeline control options
    parser.add_argument("--stage", choices=["download", "preprocess", "create_training"], 
                       help="Run only specific stage")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                       help="Skip already completed stages")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue pipeline even if a stage fails")
    parser.add_argument("--reset", action="store_true", help="Reset pipeline state")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MasterDataPipeline(data_dir=args.data_dir)
    
    # Handle special commands
    if args.reset:
        pipeline.reset_pipeline()
        return
    
    if args.status:
        logger.info("Pipeline Status:")
        logger.info(f"  Completed stages: {pipeline.state['completed_stages']}")
        logger.info(f"  Last run: {pipeline.state.get('last_run', 'Never')}")
        logger.info(f"  Errors: {len(pipeline.state['errors'])}")
        pipeline.show_data_statistics()
        return
    
    # Run pipeline
    kwargs = {
        "include_large": args.include_large,
        "workers": args.workers,
        "max_samples": args.max_samples,
        "skip_completed": args.skip_completed,
        "continue_on_error": args.continue_on_error
    }
    
    if args.stage:
        # Run single stage
        success = pipeline.run_stage(args.stage, **kwargs)
        sys.exit(0 if success else 1)
    else:
        # Run full pipeline
        success = pipeline.run_full_pipeline(**kwargs)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
