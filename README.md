# Whisper Model Comparison Tool

A comprehensive tool for comparing different Whisper model sizes on YouTube video transcription tasks. This tool helps you evaluate the trade-offs between transcription speed, accuracy, and resource usage across all available Whisper models.

## Features

- **Multi-Model Support**: Tests all Whisper model sizes (tiny, base, small, medium, large)
- **Cross-Platform**: Compatible with both MLX (Apple Silicon) and OpenAI Whisper (CUDA/CPU)
- **Comprehensive Metrics**: Provides processing time, accuracy comparison, Word Error Rate (WER), and similarity scores
- **Multiple Output Formats**: Generates JSON reports and individual SRT subtitle files
- **Quality Analysis**: Automatically determines the best speed/accuracy trade-offs

## Installation

### 1. Clone and Navigate
```bash
cd whisperModelComparison
```

### 2. Install Dependencies

#### For Apple Silicon (M1/M2/M3) - Recommended:
```bash
pip install mlx-whisper yt-dlp ffmpeg-python
```

#### For CUDA/CPU Systems:
```bash
pip install openai-whisper yt-dlp ffmpeg-python
```

#### Install All (the tool will auto-detect the best implementation):
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

### Basic Usage
```bash
python compare_models.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Advanced Usage
```bash
# Specify output directory
python compare_models.py "https://youtu.be/VIDEO_ID" --output my_results/

# Force specific implementation
python compare_models.py "https://youtu.be/VIDEO_ID" --implementation mlx

# Quiet mode (less console output)
python compare_models.py "https://youtu.be/VIDEO_ID" --quiet

# Use environment variable to force implementation
WHISPER_TYPE=openai python compare_models.py "https://youtu.be/VIDEO_ID"
```

## Output

The tool generates several types of output:

### 1. Console Report
Real-time progress and a comprehensive comparison table showing:
- Processing time for each model
- Word and character counts
- Similarity scores relative to the best model
- Word Error Rate (WER)
- Speed vs accuracy analysis

### 2. JSON Report
Detailed machine-readable results saved as `comparison_report_YYYYMMDD_HHMMSS.json` containing:
- Complete transcription results
- Timing information
- Quality metrics
- Raw Whisper outputs

### 3. SRT Subtitle Files
Individual subtitle files for each successful model:
- `transcription_tiny_YYYYMMDD_HHMMSS.srt`
- `transcription_base_YYYYMMDD_HHMMSS.srt`
- `transcription_small_YYYYMMDD_HHMMSS.srt`
- `transcription_medium_YYYYMMDD_HHMMSS.srt`
- `transcription_large_YYYYMMDD_HHMMSS.srt`

## Understanding the Results

### Performance Metrics

| Metric | Description |
|--------|-------------|
| **Processing Time** | Wall-clock time to complete transcription |
| **Word Count** | Number of words in the transcription |
| **Similarity** | Similarity ratio compared to the reference model (typically `large`) |
| **WER** | Word Error Rate - lower is better (0.0 = perfect match) |

### Model Characteristics

| Model | Size | Speed | Accuracy | MLX Available | Best For |
|-------|------|-------|----------|---------------|----------|
| **tiny** | ~39MB | Fastest | Lowest | ✅ | Real-time applications, previews |
| **base** | ~74MB | Very Fast | Low | ❌* | Quick drafts, live captions |
| **small** | ~244MB | Fast | Good | ❌* | Balanced speed/quality |
| **medium** | ~769MB | Moderate | Better | ✅ | Most use cases |
| **large** | ~1550MB | Slowest | Best | ✅ | High-accuracy requirements |

*\* Falls back to OpenAI Whisper when using MLX implementation*

### Speed vs Accuracy Analysis

The tool automatically identifies:
- **Fastest Model**: Shortest processing time
- **Highest Quality**: Best similarity to reference
- **Best Ratio**: Optimal balance of quality per unit time

## Implementation Details

### Auto-Detection Logic
1. Checks `WHISPER_TYPE` environment variable
2. Tries to import `mlx-whisper` (Apple Silicon optimized)
3. Falls back to `openai-whisper` (CUDA/CPU compatible)

### MLX Model Availability
- **Available in MLX**: `tiny`, `medium`, `large`
- **Not available in MLX**: `base`, `small` (automatically falls back to OpenAI Whisper)
- **Fallback behavior**: When using MLX implementation, unavailable models will use OpenAI Whisper

### Quality Comparison
- Uses the largest successful model as the reference for quality metrics
- Calculates text similarity using sequence matching algorithms
- Computes Word Error Rate using dynamic programming edit distance

### Resource Management
- Downloads audio once and reuses for all models
- Automatically cleans up temporary files
- Graceful error handling for individual model failures

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg for your platform
   brew install ffmpeg  # macOS
   sudo apt install ffmpeg  # Ubuntu/Debian
   ```

2. **No Whisper implementation found**
   ```bash
   # Install at least one implementation
   pip install mlx-whisper  # For Apple Silicon
   pip install openai-whisper  # For other systems
   ```

3. **YouTube download fails**
   - Check if the video is public and accessible
   - Update yt-dlp: `pip install --upgrade yt-dlp`

4. **CUDA out of memory**
   ```bash
   # Force CPU usage for OpenAI Whisper
   WHISPER_TYPE=openai python compare_models.py "URL"
   ```

5. **MLX model not found errors**
   - This is expected for `base` and `small` models (they don't exist in MLX format)
   - The tool automatically falls back to OpenAI Whisper for these models
   - Ensure you have `openai-whisper` installed: `pip install openai-whisper`

### Performance Tips

- **Apple Silicon**: Use MLX implementation for best performance
- **NVIDIA GPU**: Ensure CUDA-compatible PyTorch is installed
- **Large videos**: Consider shorter clips for initial testing
- **Memory constrained**: Start with smaller models only

## Example Output

```
WHISPER MODEL COMPARISON REPORT
================================================================================
URL: https://www.youtube.com/watch?v=example
Implementation: mlx (with OpenAI fallback)
Reference Model: large
Timestamp: 2024-01-15T10:30:45

PERFORMANCE COMPARISON
--------------------------------------------------------------------------------
Model    Success  Time(s)  Words    Chars    Similarity WER     Implementation
--------------------------------------------------------------------------------
tiny     ✓        12.3     1247     6891     0.856      0.144   MLX
base     ✓        18.7     1302     7234     0.923      0.077   OpenAI (fallback)
small    ✓        31.2     1334     7456     0.967      0.033   OpenAI (fallback)
medium   ✓        52.8     1345     7523     0.989      0.011   MLX
large    ✓        89.4     1348     7534     1.000      0.000   MLX

SPEED vs ACCURACY ANALYSIS
--------------------------------------------------------------------------------
Fastest: tiny (12.3s)
Highest Quality: large (similarity: 1.000)
Best Quality/Speed Ratio: small (ratio: 0.0310)

Note: base and small models used OpenAI Whisper fallback (MLX versions not available)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the original model
- [MLX Whisper](https://github.com/ml-explore/mlx-examples) for Apple Silicon optimization
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube audio extraction 