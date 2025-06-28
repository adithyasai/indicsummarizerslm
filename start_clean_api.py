#!/usr/bin/env python3
"""
Simple startup script for the Clean Telugu Summarizer API
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_dependencies():
    """Quick dependency check"""
    try:
        import fastapi, uvicorn, torch, transformers
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install -r requirements.txt")
        return False

def check_summarizer():
    """Check if the summarizer can be loaded"""
    try:
        from proper_slm_summarizer import get_proper_slm_summarizer
        summarizer = get_proper_slm_summarizer()
        return True
    except Exception as e:
        print(f"⚠️ Summarizer issue: {e}")
        print("💡 The server will still start with fallback mode")
        return False

def main():
    """Main startup function"""
    print("🚀 Telugu Summarizer - Clean API Server Startup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print("✅ Python version:", sys.version.split()[0])
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        return False
    print("✅ Dependencies satisfied")
    
    # Check summarizer
    print("\n🧠 Checking summarizer...")
    summarizer_ok = check_summarizer()
    if summarizer_ok:
        print("✅ Summarizer ready")
    
    # Start server
    print("\n🌐 Starting Clean API Server...")
    print("📍 URL: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("❤️ Health: http://localhost:8000/health")
    print("\n💡 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run
        from clean_api_server import app
        import uvicorn
        
        uvicorn.run(
            "clean_api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1)
