#!/usr/bin/env python3
"""
Whisper Model Comparison Tool

This script transcribes a YouTube video using all available Whisper model sizes
and provides detailed comparison metrics including transcription quality,
processing time, and resource usage.

Compatible with both MLX (Apple Silicon) and OpenAI Whisper (CUDA/CPU).
"""

import os
import sys
import time
import json
import argparse
import uuid
import yt_dlp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import difflib

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

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

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
    return mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_name)

def transcribe_with_openai(audio_path: str, model_size: str) -> Dict[str, Any]:
    """Transcribe audio using OpenAI Whisper (CUDA/CPU compatible)."""
    import whisper
    print(f"  Using OpenAI Whisper ({model_size} model)...")
    model = whisper.load_model(model_size)
    return model.transcribe(audio_path, fp16=False)  # fp16=False for CPU compatibility

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
        return result["text"].strip()
    
    # Extract from segments if available
    if "segments" in result:
        return " ".join([segment["text"].strip() for segment in result["segments"]])
    
    return ""

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using difflib."""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts (lowercase, remove extra spaces)
    text1_norm = " ".join(text1.lower().split())
    text2_norm = " ".join(text2.lower().split())
    
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, text1_norm, text2_norm).ratio()
    return similarity

def generate_srt_content(result: Dict[str, Any]) -> str:
    """Generate SRT content from transcription result."""
    if "error" in result or "segments" not in result:
        return ""
    
    srt_content = []
    for i, segment in enumerate(result['segments']):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        srt_content.append(f"{i + 1}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(f"{text}\n")
    
    return "\n".join(srt_content)

def calculate_word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis."""
    if not reference or not hypothesis:
        return 1.0
    
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Dynamic programming to calculate edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # WER = edit_distance / reference_length
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    return dp[m][n] / len(ref_words)

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
        
        # Store detailed results
        results[model_size] = {
            "success": success,
            "processing_time": processing_time,
            "transcript_text": transcript_text,
            "word_count": len(transcript_text.split()) if transcript_text else 0,
            "char_count": len(transcript_text) if transcript_text else 0,
            "raw_result": result if success else None,
            "srt_content": generate_srt_content(result) if success else "",
            "implementation_used": actual_implementation
        }
        
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Success: {success}")
        print(f"  Word count: {results[model_size]['word_count']}")
        print(f"  Character count: {results[model_size]['char_count']}")
    
    # Step 3: Calculate comparison metrics
    print("\nStep 3: Calculating comparison metrics...")
    print("="*60)
    
    # Use the largest successful model as reference for quality comparison
    reference_model = None
    for model in reversed(model_sizes):  # Start from largest
        if results[model]["success"] and results[model]["transcript_text"]:
            reference_model = model
            break
    
    if reference_model:
        reference_text = all_transcripts[reference_model]
        print(f"Using {reference_model} model as reference for quality comparison")
        
        for model_size in model_sizes:
            if results[model_size]["success"] and model_size != reference_model:
                similarity = calculate_similarity(reference_text, all_transcripts[model_size])
                wer = calculate_word_error_rate(reference_text, all_transcripts[model_size])
                
                results[model_size]["similarity_to_reference"] = similarity
                results[model_size]["word_error_rate"] = wer
            else:
                results[model_size]["similarity_to_reference"] = 1.0 if model_size == reference_model else 0.0
                results[model_size]["word_error_rate"] = 0.0 if model_size == reference_model else 1.0
    
    # Clean up temporary audio file
    try:
        os.remove(audio_path)
        print(f"\nCleaned up temporary file: {audio_path}")
    except OSError:
        print(f"\nWarning: Could not remove temporary file: {audio_path}")
    
    # Create comparison summary
    comparison_summary = {
        "url": url,
        "implementation": implementation,
        "reference_model": reference_model,
        "timestamp": datetime.now().isoformat(),
        "models": results
    }
    
    return comparison_summary

def print_comparison_report(comparison_data: Dict[str, Any]):
    """Print a detailed comparison report to console."""
    print("\n" + "="*80)
    print("WHISPER MODEL COMPARISON REPORT")
    print("="*80)
    
    print(f"URL: {comparison_data['url']}")
    print(f"Implementation: {comparison_data['implementation']}")
    print(f"Reference Model: {comparison_data['reference_model']}")
    print(f"Timestamp: {comparison_data['timestamp']}")
    
    print("\n" + "-"*95)
    print("PERFORMANCE COMPARISON")
    print("-"*95)
    
    # Performance table header
    print(f"{'Model':<8} {'Success':<8} {'Time(s)':<8} {'Words':<8} {'Chars':<8} {'Similarity':<10} {'WER':<8} {'Implementation':<15}")
    print("-"*95)
    
    for model_size, data in comparison_data['models'].items():
        success = "✓" if data['success'] else "✗"
        time_str = f"{data['processing_time']:.1f}" if data['success'] else "N/A"
        words = data['word_count'] if data['success'] else 0
        chars = data['char_count'] if data['success'] else 0
        similarity = f"{data.get('similarity_to_reference', 0):.3f}" if data['success'] else "N/A"
        wer = f"{data.get('word_error_rate', 1):.3f}" if data['success'] else "N/A"
        impl = data.get('implementation_used', 'unknown')
        
        print(f"{model_size:<8} {success:<8} {time_str:<8} {words:<8} {chars:<8} {similarity:<10} {wer:<8} {impl:<15}")
    
    print("\n" + "-"*95)
    print("SPEED vs ACCURACY ANALYSIS")
    print("-"*95)
    
    successful_models = [(k, v) for k, v in comparison_data['models'].items() if v['success']]
    if len(successful_models) > 1:
        # Sort by processing time
        by_speed = sorted(successful_models, key=lambda x: x[1]['processing_time'])
        print(f"Fastest: {by_speed[0][0]} ({by_speed[0][1]['processing_time']:.1f}s)")
        
        # Sort by similarity (quality)
        by_quality = sorted(successful_models, key=lambda x: x[1].get('similarity_to_reference', 0), reverse=True)
        print(f"Highest Quality: {by_quality[0][0]} (similarity: {by_quality[0][1].get('similarity_to_reference', 0):.3f})")
        
        # Speed/Quality ratio (higher is better)
        for model, data in successful_models:
            if data['processing_time'] > 0:
                quality_speed_ratio = data.get('similarity_to_reference', 0) / data['processing_time']
                data['quality_speed_ratio'] = quality_speed_ratio
        
        by_ratio = sorted(successful_models, key=lambda x: x[1].get('quality_speed_ratio', 0), reverse=True)
        print(f"Best Quality/Speed Ratio: {by_ratio[0][0]} (ratio: {by_ratio[0][1].get('quality_speed_ratio', 0):.4f})")

def save_results(comparison_data: Dict[str, Any], output_dir: str):
    """Save comparison results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    json_path = os.path.join(output_dir, f"comparison_report_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed JSON report saved to: {json_path}")
    
    # Save individual SRT files
    for model_size, data in comparison_data['models'].items():
        if data['success'] and data['srt_content']:
            srt_path = os.path.join(output_dir, f"transcription_{model_size}_{timestamp}.srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(data['srt_content'])
            print(f"SRT file for {model_size} model saved to: {srt_path}")

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
        comparison_data = compare_whisper_models(url, args.output, args.implementation)
        
        # Print report unless quiet mode
        if not args.quiet:
            print_comparison_report(comparison_data)
        
        # Save results
        save_results(comparison_data, args.output)
        
        print(f"\nComparison complete! Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 