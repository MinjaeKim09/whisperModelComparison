#!/usr/bin/env python3
"""
Whisper Model Comparison Tool

This script transcribes a YouTube video using all available Whisper model sizes
and provides detailed comparison metrics including processing time and resource usage.

Compatible with both MLX (Apple Silicon) and OpenAI Whisper (CUDA/CPU).
"""

import os
import sys
import time
import argparse
import uuid
import yt_dlp
from typing import Dict, List, Tuple, Any, Optional

def get_whisper_implementation():
    """
    Auto-detect and return the best available Whisper implementation.
    Returns: 'mlx', 'openai', or 'none'
    """
    # Check for manual override via environment variable
    forced_type = os.getenv("WHISPER_TYPE", "").lower()
    if forced_type in ["mlx", "openai"]:
        return forced_type
    
    # Try MLX-Whisper first (optimized for Apple Silicon)
    try:
        import mlx_whisper
        return "mlx"
    except ImportError:
        pass
    
    # Fallback to OpenAI Whisper
    try:
        import whisper
        return "openai"
    except ImportError:
        pass
    
    return "none"



def transcribe_with_mlx(audio_path: str, model_size: str) -> Dict[str, Any]:
    """Transcribe audio using MLX-Whisper (Apple Silicon optimized)."""
    import mlx_whisper
    print(f"  Using MLX-Whisper ({model_size} model)...")
    
    # Map model sizes to correct MLX model names
    mlx_model_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "base",  # Use OpenAI model name for unavailable MLX models
        "small": "small",  # Use OpenAI model name for unavailable MLX models  
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-mlx"
    }
    
    model_name = mlx_model_map.get(model_size, model_size)
    # Request just text output without segments/timestamps
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_name)
    
    # Extract just the text and return a simplified result
    if "text" in result:
        text = result["text"]
        if isinstance(text, str):
            return {"text": text.strip()}
        else:
            return {"text": str(text).strip()}
    else:
        return result

def transcribe_with_openai(audio_path: str, model_size: str) -> Dict[str, Any]:
    """Transcribe audio using OpenAI Whisper (CUDA/CPU compatible)."""
    import whisper
    print(f"  Using OpenAI Whisper ({model_size} model)...")
    model = whisper.load_model(model_size)
    # Request just text output without segments/timestamps
    result = model.transcribe(audio_path, fp16=False)  # fp16=False for CPU compatibility
    
    # Extract just the text and return a simplified result
    if "text" in result:
        text = result["text"]
        if isinstance(text, str):
            return {"text": text.strip()}
        else:
            return {"text": str(text).strip()}
    else:
        return result

def download_youtube_audio(url: str, output_dir: str) -> str:
    """Download audio from YouTube URL and return the path to the WAV file."""
    request_id = str(uuid.uuid4())
    output_filename_base = f'audio_{request_id}'
    output_path_template = os.path.join(output_dir, f'{output_filename_base}.%(ext)s')
    final_wav_path = os.path.join(output_dir, f'{output_filename_base}.wav')

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': output_path_template,
        'quiet': True,
        'noplaylist': True,
    }

    print(f"Downloading audio from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    if not os.path.exists(final_wav_path):
        raise FileNotFoundError(f"Expected audio file not found at {final_wav_path}")
    
    print(f"Audio saved to: {final_wav_path}")
    return final_wav_path

def transcribe_with_model(audio_path: str, model_size: str, implementation: str) -> Tuple[Dict[str, Any], float, bool]:
    """
    Transcribe audio with specified model and return results, processing time, and success status.
    """
    start_time = time.time()
    
    try:
        if implementation == "mlx":
            result = transcribe_with_mlx(audio_path, model_size)
        elif implementation == "openai":
            result = transcribe_with_openai(audio_path, model_size)
        else:
            raise ValueError(f"Unsupported implementation: {implementation}")
        
        processing_time = time.time() - start_time
        return result, processing_time, True
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"  ERROR with {model_size} model: {e}")
        return {"error": str(e)}, processing_time, False

def extract_text_from_result(result: Dict[str, Any]) -> str:
    """Extract plain text from transcription result."""
    if "error" in result:
        return ""
    
    if "text" in result:
        text = result["text"]
        if isinstance(text, str):
            return text.strip()
        else:
            return str(text).strip()
    
    # Fallback: Extract from segments if available (for backward compatibility)
    if "segments" in result:
        return " ".join([segment["text"].strip() for segment in result["segments"]])
    
    return ""





def compare_whisper_models(url: str, output_dir: str, implementation: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare all Whisper model sizes on a given YouTube URL.
    """
    # Available model sizes (from smallest to largest)
    model_sizes = ["tiny", "base", "small", "medium", "large"]
    
    # MLX models that are confirmed to be available
    available_mlx_models = {"tiny", "medium", "large"}  # base and small don't exist in MLX format
    
    # Auto-detect implementation if not specified
    if implementation is None:
        detected_implementation = get_whisper_implementation()
        if detected_implementation == "none":
            raise RuntimeError("No Whisper implementation found. Please install mlx-whisper or openai-whisper.")
        implementation = detected_implementation
    
    # Type assertion for linter
    assert implementation is not None, "Implementation should not be None at this point"
    
    print(f"Using Whisper implementation: {implementation}")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download audio once for all models
    print("Step 1: Downloading audio...")
    audio_path = download_youtube_audio(url, output_dir)
    
    # Store results for each model
    results = {}
    all_transcripts = {}
    
    print("\nStep 2: Transcribing with different models...")
    print("="*60)
    
    for model_size in model_sizes:
        print(f"\nProcessing with {model_size} model...")
        
        # Skip unavailable MLX models and suggest alternatives
        if implementation == "mlx" and model_size not in available_mlx_models:
            print(f"  WARNING: {model_size} model not available in MLX format. Falling back to OpenAI Whisper...")
            # Check if OpenAI Whisper is available for fallback
            try:
                import whisper
                result, processing_time, success = transcribe_with_model(audio_path, model_size, "openai")
            except ImportError:
                print(f"  ERROR: OpenAI Whisper not available for fallback. Install with: pip install openai-whisper")
                result, processing_time, success = {"error": "OpenAI Whisper not available for fallback"}, 0.0, False
        else:
            result, processing_time, success = transcribe_with_model(audio_path, model_size, implementation)
        
        # Extract text for comparison
        transcript_text = extract_text_from_result(result) if success else ""
        all_transcripts[model_size] = transcript_text
        
        # Determine which implementation was actually used
        actual_implementation = implementation
        if implementation == "mlx" and model_size not in available_mlx_models:
            actual_implementation = "openai (fallback)"
        
        # Store minimal results
        results[model_size] = {
            "success": success,
            "processing_time": processing_time,
            "transcript_text": transcript_text,
            "implementation_used": actual_implementation
        }
        
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Success: {success}")
    

    
    # Clean up temporary audio file
    try:
        os.remove(audio_path)
        print(f"\nCleaned up temporary file: {audio_path}")
    except OSError:
        print(f"\nWarning: Could not remove temporary file: {audio_path}")
    
    return results

def print_timing_report(results: Dict[str, Any]):
    """Print a simple timing report to console."""
    print("\n" + "="*40)
    print("PROCESSING TIME COMPARISON")
    print("="*40)
    
    # Performance table header
    print(f"{'Model':<8} {'Success':<8} {'Time(s)':<8}")
    print("-"*25)
    
    for model_size, data in results.items():
        success = "✓" if data['success'] else "✗"
        time_str = f"{data['processing_time']:.1f}" if data['success'] else "N/A"
        
        print(f"{model_size:<8} {success:<8} {time_str:<8}")

def save_text_files(results: Dict[str, Any], output_dir: str):
    """Save transcription text files."""
    # Save individual text files (simple names, overwritten each run)
    for model_size, data in results.items():
        if data['success'] and data['transcript_text']:
            txt_path = os.path.join(output_dir, f"{model_size}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(data['transcript_text'])
            print(f"Text file for {model_size} model saved to: {txt_path}")

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "results")
    
    parser = argparse.ArgumentParser(
        description="Compare Whisper model sizes on YouTube video transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_models.py
  python compare_models.py --output results/
  python compare_models.py --implementation mlx
  WHISPER_TYPE=openai python compare_models.py
        """
    )
    
    parser.add_argument("--output", "-o", default=default_output, 
                       help="Output directory for results (default: results/ in script directory)")
    parser.add_argument("--implementation", choices=["mlx", "openai"], 
                       help="Force specific Whisper implementation (auto-detect if not specified)")
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="Suppress detailed console output")
    
    args = parser.parse_args()
    
    # If user provided a relative path, make it relative to script directory
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    # Prompt user for YouTube URL
    print("Whisper Model Comparison Tool")
    print("="*40)
    while True:
        url = input("Please enter a YouTube URL: ").strip()
        if url:
            # Basic validation - check if it looks like a YouTube URL
            if "youtube.com" in url or "youtu.be" in url:
                break
            else:
                print("Please enter a valid YouTube URL (containing 'youtube.com' or 'youtu.be')")
        else:
            print("URL cannot be empty. Please try again.")
    
    try:
        # Run comparison
        results = compare_whisper_models(url, args.output, args.implementation)
        
        # Print timing report unless quiet mode
        if not args.quiet:
            print_timing_report(results)
        
        # Save text files
        save_text_files(results, args.output)
        
        print(f"\nComparison complete! Text files saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 