#!/bin/bash
# SVOD Installation Script
# Simple installer for end users

set -e

echo "========================================"
echo "SVOD - Smart Video Orientation Detector"
echo "Installation Script"
echo "========================================"

# Check Python version
echo "Checking Python version..."
if ! python3 --version >/dev/null 2>&1; then
    echo "Error: Python 3 is required but not found."
    echo "Please install Python 3.11 or later from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
REQUIRED_VERSION="3.11"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "Error: Python $REQUIRED_VERSION or later is required."
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION found"

# Check pip
echo "Checking pip..."
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "Error: pip is required but not found."
    exit 1
fi

echo "✓ pip found"

# Install SVOD
echo "Installing SVOD..."
python3 -m pip install --user .

# Test installation
echo "Testing installation..."
if python3 -c "import video_orientation_detector; print('✓ Import successful')"; then
    echo ""
    echo "========================================"
    echo "Installation completed successfully!"
    echo "========================================"
    echo ""
    echo "Usage:"
    echo "  svod video.mp4                    # Analyze single video"
    echo "  svod /path/to/videos --batch      # Analyze folder"
    echo "  svod --help                       # Show all options"
    echo ""
    echo "For more information, see README.md"
else
    echo "Error: Installation test failed"
    exit 1
fi