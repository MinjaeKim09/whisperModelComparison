#!/usr/bin/env python3
"""
Example script demonstrating the Whisper Model Comparison Tool

This script runs a comparison on a sample YouTube video to demonstrate
the capabilities of the comparison tool.
"""

import os
import sys
from compare_models import compare_whisper_models, print_comparison_report, save_results

def run_example():
    """Run an example comparison on a sample YouTube video."""
    
    # Example YouTube URLs (short videos for quick testing)
    example_urls = [
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # "Me at the zoo" - first YouTube video (19s)
        "https://www.youtube.com/watch?v=9bZkp7q19f0",  # PSY - Gangnam Style (short clip)
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
    ]
    
    # Use the first URL as default
    url = example_urls[0]
    
    print("Whisper Model Comparison Tool - Example Run")
    print("=" * 50)
    print(f"This example will compare all Whisper models on:")
    print(f"URL: {url}")
    print(f"Note: This is a short video to demonstrate the tool quickly.")
    print()
    
    # Ask user if they want to proceed or use a different URL
    response = input("Press Enter to continue with this URL, or enter a different YouTube URL: ").strip()
    if response:
        url = response
    
    # Set output directory
    output_dir = "./example_results"
    
    try:
        print(f"\nStarting comparison...")
        print(f"Output directory: {output_dir}")
        print("This may take several minutes depending on the video length and your hardware.\n")
        
        # Run the comparison
        comparison_data = compare_whisper_models(url, output_dir)
        
        # Print detailed report
        print_comparison_report(comparison_data)
        
        # Save results
        save_results(comparison_data, output_dir)
        
        print("\n" + "="*50)
        print("Example completed successfully!")
        print(f"Check the '{output_dir}' directory for detailed results.")
        print("\nFiles generated:")
        print("- JSON report with detailed metrics")
        print("- Individual SRT files for each model")
        print("\nTry running with your own YouTube URLs:")
        print("python compare_models.py 'YOUR_YOUTUBE_URL'")
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running example: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have installed the requirements: pip install -r requirements.txt")
        print("2. Ensure FFmpeg is installed on your system")
        print("3. Check your internet connection")
        print("4. Try with a different YouTube URL")
        sys.exit(1)

if __name__ == "__main__":
    run_example() 