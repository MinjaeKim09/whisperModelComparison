#!/bin/bash

# Whisper Model Comparison Tool - Shell Script Wrapper
# This script provides an easy way to run the model comparison tool

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Whisper Model Comparison Tool${NC}"
echo "=============================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if URL is provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: $0 <YouTube_URL> [options]${NC}"
    echo ""
    echo "Examples:"
    echo "  $0 'https://www.youtube.com/watch?v=VIDEO_ID'"
    echo "  $0 'https://youtu.be/VIDEO_ID' --output my_results/"
    echo "  $0 'https://youtu.be/VIDEO_ID' --implementation mlx"
    echo ""
    echo "Options:"
    echo "  --output, -o DIR     Output directory (default: ./results)"
    echo "  --implementation     Force implementation (mlx or openai)"
    echo "  --quiet, -q          Suppress detailed output"
    echo "  --test               Run setup test"
    echo "  --example            Run example comparison"
    echo ""
    exit 1
fi

# Handle special commands
if [ "$1" = "--test" ] || [ "$1" = "test" ]; then
    echo -e "${BLUE}Running setup test...${NC}"
    python3 test_setup.py
    exit $?
fi

if [ "$1" = "--example" ] || [ "$1" = "example" ]; then
    echo -e "${BLUE}Running example comparison...${NC}"
    python3 run_example.py
    exit $?
fi

# Check if the main script exists
if [ ! -f "compare_models.py" ]; then
    echo -e "${RED}Error: compare_models.py not found in current directory${NC}"
    echo "Please run this script from the whisperModelComparison directory"
    exit 1
fi

# Run the comparison tool with all provided arguments
echo -e "${GREEN}Starting Whisper model comparison...${NC}"
echo "URL: $1"
echo ""

python3 compare_models.py "$@"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Comparison completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}✗ Comparison failed. Check the error messages above.${NC}"
    exit 1
fi 