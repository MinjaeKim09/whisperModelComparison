#!/usr/bin/env python3
"""
Test script to verify that the Whisper Model Comparison Tool is set up correctly.
This script checks for required dependencies and basic functionality.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    # Test core dependencies
    try:
        import yt_dlp
        print("✓ yt-dlp is available")
    except ImportError:
        print("✗ yt-dlp is not installed. Run: pip install yt-dlp")
        return False
    
    # Test Whisper implementations
    mlx_available = False
    openai_available = False
    
    try:
        import mlx_whisper
        print("✓ MLX-Whisper is available (Apple Silicon optimized)")
        mlx_available = True
    except ImportError:
        print("- MLX-Whisper not available (install with: pip install mlx-whisper)")
    
    try:
        import whisper
        print("✓ OpenAI-Whisper is available (CUDA/CPU compatible)")
        openai_available = True
    except ImportError:
        print("- OpenAI-Whisper not available (install with: pip install openai-whisper)")
    
    if not mlx_available and not openai_available:
        print("✗ No Whisper implementation found! Install at least one:")
        print("  - For Apple Silicon: pip install mlx-whisper")
        print("  - For CUDA/CPU: pip install openai-whisper")
        return False
    
    # Test other dependencies
    try:
        import json
        import argparse
        import uuid
        import time
        import difflib
        from datetime import datetime
        from pathlib import Path
        print("✓ All core Python modules are available")
    except ImportError as e:
        print(f"✗ Missing core Python module: {e}")
        return False
    
    return True

def test_ffmpeg():
    """Test if FFmpeg is available."""
    print("\nTesting FFmpeg...")
    
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            print("✓ FFmpeg is available")
            return True
        else:
            print("✗ FFmpeg is not working properly")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg is not installed or not in PATH")
        print("  Install FFmpeg:")
        print("  - macOS: brew install ffmpeg")
        print("  - Ubuntu/Debian: sudo apt install ffmpeg")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        return False
    except subprocess.TimeoutExpired:
        print("✗ FFmpeg command timed out")
        return False

def test_compare_models_import():
    """Test if the main comparison module can be imported."""
    print("\nTesting main module...")
    
    try:
        from compare_models import get_whisper_implementation, compare_whisper_models
        print("✓ Main comparison module imports successfully")
        
        # Test implementation detection
        impl = get_whisper_implementation()
        if impl != "none":
            print(f"✓ Detected Whisper implementation: {impl}")
            return True
        else:
            print("✗ No Whisper implementation detected")
            return False
            
    except ImportError as e:
        print(f"✗ Cannot import main module: {e}")
        return False

def main():
    """Run all setup tests."""
    print("Whisper Model Comparison Tool - Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    if not test_imports():
        all_tests_passed = False
    
    if not test_ffmpeg():
        all_tests_passed = False
    
    if not test_compare_models_import():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED!")
        print("The Whisper Model Comparison Tool is ready to use.")
        print("\nTo get started:")
        print('python compare_models.py "https://www.youtube.com/watch?v=VIDEO_ID"')
        print("or")
        print("python run_example.py")
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please install missing dependencies before using the tool.")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 