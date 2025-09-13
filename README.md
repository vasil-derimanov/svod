# Smart Video Orientation Detector (SVOD)

🎥 **AI-Powered Video Orientation Analysis** | **YOLOv8 + Face Detection** | **Cross-Platform**

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/vasil-derimanov/svod/actions/workflows/cross-platform-test.yml/badge.svg)](https://github.com/vasil-derimanov/svod/actions)

## 📋 Overview

SVOD automatically detects video orientation using advanced AI techniques:
- **Face Detection** - Primary orientation indicator using DNN face detector
- **Body Detection** - YOLOv8 analysis for person detection
- **Facial Landmarks** - Precise orientation analysis using LBF landmark detection
- **Cross-Platform** - Optimized for Windows, Linux, and Apple Silicon (M1/M2/M3)
- **Batch Processing** - Process entire folders with comprehensive reports
- **Security Hardened** - Input validation, resource limits, and safe defaults

## 🚀 Key Features

- ✅ **Single Video Analysis** - Process individual videos with detailed reports
- ✅ **Batch Processing** - Process folders recursively with summary reports
- ✅ **Real-time Display** - Optional live preview during analysis
- ✅ **Flexible Output** - Save annotated videos and detailed reports
- ✅ **Time-Limited Analysis** - Configurable processing time for large datasets
- ✅ **Validation Mode** - Compare results against reference data
- ✅ **Automatic Setup** - All dependencies installed automatically
- ✅ **Security Features** - Input sanitization and resource protection
- ✅ **Cross-Platform** - Windows, Linux, macOS support

## 📦 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/vasil-derimanov/svod.git
cd svod
```

2. **Run the script - automatic setup!**
```bash
python video_orientation_detector.py your_video.mp4
```

**Everything is installed automatically:**
- Python packages (opencv-contrib-python, numpy, ultralytics, etc.)
- AI models (YOLOv8, face detector, facial landmarks)
- No manual configuration required!

### Basic Usage

```bash
# Analyze single video
python video_orientation_detector.py video.mp4

# Process folder of videos
python video_orientation_detector.py /path/to/videos --batch

# Save annotated output
python video_orientation_detector.py video.mp4 -o corrected.mp4
```

## 💻 Usage Examples

### Single Video Analysis

```bash
# Basic analysis with display
python video_orientation_detector.py video.mp4

# Process without display (faster)
python video_orientation_detector.py video.mp4 --no-display

# Analyze only first 10 seconds
python video_orientation_detector.py video.mp4 --time-limit 10

# Higher confidence threshold
python video_orientation_detector.py video.mp4 -c 0.7

# Save annotated video output
python video_orientation_detector.py video.mp4 -o annotated.mp4
```

### Batch Processing

```bash
# Process all videos in folder
python video_orientation_detector.py /path/to/videos --batch

# Include subfolders recursively
python video_orientation_detector.py /path/to/videos --batch -r

# Save detailed report
python video_orientation_detector.py /path/to/videos --batch --report results.json

# Limit processing time per video
python video_orientation_detector.py /path/to/videos --batch --time-limit 15

# Security: Limit files and depth
python video_orientation_detector.py /path/to/videos --batch --max-files 500 --max-depth 5
```

### Advanced Options

```bash
# Use reference file for validation
python video_orientation_detector.py folder --batch --reference orientations.csv

# Combined options for large datasets
python video_orientation_detector.py /videos --batch -r --time-limit 30 --report detailed.json -c 0.8
```

## ⚙️ Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `path` | | Video file or folder path | Required |
| `--output` | `-o` | Save annotated video | None |
| `--no-display` | | Process without preview | False |
| `--confidence` | `-c` | Detection threshold (0.0-1.0) | 0.5 |
| `--time-limit` | `-t` | Max analysis time (seconds) | 30 |
| `--no-time-limit` | | Analyze entire video | False |
| `--batch` | | Enable batch processing | False |
| `--recursive` | `-r` | Process subfolders | False |
| `--report` | | Save batch report (JSON/CSV) | None |
| `--reference` | | Reference file for validation | None |
| `--max-files` | | Max files in batch mode | 1000 |
| `--max-depth` | | Max directory depth | 10 |
| `--version` | | Show version information | |

## 🔧 System Requirements

### Minimum Requirements
- **Python**: 3.11 - 3.12 (3.13+ not supported)
- **RAM**: 4GB+ for optimal performance
- **Storage**: 100MB free space for models
- **Internet**: Required for initial model downloads

### Platform Support

#### Windows
- ✅ Full support with all features
- ✅ Automatic model downloads
- ✅ GPU acceleration available

#### Linux / WSL
- ✅ Full support with all features
- ✅ OpenVINO optimization available
- ✅ Automatic model downloads

#### macOS / Apple Silicon
- ✅ Core features fully supported
- ⚠️ OpenVINO limited (optional enhancement)
- ✅ Automatic fallbacks for compatibility

### Dependencies (Auto-Installed)

**Core Dependencies:**
- `opencv-contrib-python` - Computer vision and face detection
- `numpy` - Mathematical operations
- `ultralytics` - YOLOv8 object detection (required)
- `torch` - PyTorch for model operations
- `onnx` - Model format conversion
- `tqdm` - Progress bars

**Platform-Specific:**
- `openvino` - Intel optimization (Linux/Windows)
- `openvino-dev` - Model tools (Linux/Windows)

## 🛠️ Troubleshooting

### Common Issues

**"YOLOv8 not available" Error:**
```bash
# Install ultralytics manually
pip install ultralytics

# For macOS NumPy issues
pip install "numpy<2.0"
```

**Model Download Failures:**
- Check internet connectivity
- Verify firewall/proxy settings
- Models will retry automatically on next run

**Performance Issues:**
- Use `--time-limit` for large videos
- Enable `--no-display` for faster processing
- Close other GPU-intensive applications

**Cross-Platform Issues:**
- Apple Silicon: MobileNet models optional
- WSL: Ensure OpenVINO is installed
- Windows: Check antivirus exclusions

### Manual Model Downloads

If automatic downloads fail, download models manually:

```bash
# YOLOv8 (auto-downloaded via ultralytics)
# Face detection models
curl -O https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -O https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

# Facial landmarks
curl -O https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml

# COCO class names
curl -O https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

# MobileNet models (optional enhancement)
curl -O https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/mobilenet-v2-pytorch/FP32/mobilenet-v2.xml
curl -O https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/mobilenet-v2-pytorch/FP32/mobilenet-v2.bin
```

### Error Codes

- **Exit 0**: Success
- **Exit 1**: Critical model files missing
- **Exit 2**: Invalid input parameters
- **Exit 3**: Video processing error

## 🔍 Detection Methods

### Primary Detection: Face Analysis
- Uses OpenCV DNN face detector
- Analyzes face orientation and positioning
- Most accurate for videos with visible faces

### Secondary Detection: Body Analysis
- YOLOv8 object detection for people
- Backup method when faces aren't visible
- Useful for wide shots and group scenes

### Enhanced Detection: MobileNet Classification
- Optional enhancement for improved accuracy
- Cross-platform fallback support
- Automatic activation when available

### Smart Analysis Logic
- **Balanced Weighting**: Faces and bodies contribute equally (50/50)
- **Face-Only Detection**: Special handling for high face-density videos
- **Mobile Portrait Override**: Automatic detection of mobile videos
- **Time-Sampling**: Distributed analysis across video segments

## 📊 Output Formats

### Console Output
```
VIDEO ORIENTATION ANALYSIS RESULTS
==============================================================
CORRECT
Confidence: 87.3%
Recommendation: No action needed

[STATS] Frame Analysis:
  • Total frames analyzed: 450
  • Frames with humans: 387
  • Correct orientation: 92.1%
  • Incorrect orientation: 7.9%

[TIMER] Time Analysis:
  • Video duration: 15.0s
  • Analyzed duration: 15.0s
  • Analysis coverage: 100.0%
==============================================================
```

### Batch Reports
- **JSON Format**: Complete analysis data with detection details
- **CSV Format**: Tabular results for spreadsheet analysis
- **Annotated Videos**: Visual overlays showing detection results

### Validation Mode
Compare results against reference orientation data:
```bash
python video_orientation_detector.py folder --batch --reference reference.csv
```

## 🧪 Testing & Quality Assurance

### Automated Testing
```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run CI tests (no coverage)
make test-ci
```

### Clean Environment Testing
```bash
# PowerShell cleanup
.\cleanup.ps1

# Python cleanup
python cleanup.py
```

### Test Coverage
- ✅ Unit tests for core detection logic
- ✅ Integration tests for CLI functionality
- ✅ Batch processing validation
- ✅ Cross-platform compatibility
- ✅ Security hardening verification

## 🔒 Security Features

### Input Validation
- Path sanitization and length limits
- Dangerous character detection
- Directory traversal protection
- File type validation

### Resource Protection
- Configurable time limits per video
- Batch processing file count limits
- Directory depth restrictions
- Memory usage monitoring

### Safe Defaults
- Conservative confidence thresholds
- Reasonable processing timeouts
- Automatic resource checks

## 📈 Performance & Accuracy

### Detection Accuracy
- **Face Detection**: 85-95% accuracy for videos with faces
- **Body Detection**: 75-85% accuracy as backup method
- **Combined Analysis**: 90%+ overall accuracy with balanced weighting

### Processing Speed
- **Single Video**: 5-30 seconds depending on length
- **Batch Processing**: 2-10x faster with `--no-display`
- **Time Limits**: Configurable analysis duration
- **GPU Acceleration**: Automatic when available

### System Resources
- **RAM**: 2-4GB typical usage
- **CPU**: Multi-core optimized
- **Storage**: 100MB for models + output files

## 📝 API Documentation

### Core Classes

#### `OrientationDetector`
Main detection class with comprehensive video analysis capabilities.

**Key Methods:**
- `process_video()` - Analyze single video file
- `process_folder()` - Batch process video directory
- `validate_against_reference()` - Compare results with reference data

**Parameters:**
- `confidence_threshold` (float): Detection confidence cutoff (0.0-1.0)
- `time_limit` (float): Maximum analysis time per video in seconds

#### `VideoOrientation`
Enum representing possible orientation states:
- `CORRECT` - Video is properly oriented
- `INCORRECT` - Video needs rotation
- `UNCERTAIN` - Cannot determine orientation confidently

### Configuration Files

#### `pyproject.toml`
```toml
[tool.black]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

#### `requirements.txt`
```
opencv-contrib-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
torch>=2.0.0
onnx>=1.14.0
tqdm>=4.65.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

## 🚀 Deployment

### PyPI Package (Planned)
```bash
# Future PyPI installation
pip install svod
```

### Docker Deployment
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python video_orientation_detector.py --version

CMD ["python", "video_orientation_detector.py"]
```

### CI/CD Integration
- GitHub Actions workflows for cross-platform testing
- Automated testing on Windows, Linux, macOS
- Coverage reporting and quality checks

## 📋 Version History

### Recent Versions
- **v4.20.0** (2025-09-13): Enhanced error handling & security hardening
- **v4.19.2** (2025-01-21): YOLOv8 mandatory, NumPy compatibility fixes
- **v4.19.0** (2025-01-21): Face-only rotation detection, zero false positives
- **v4.17.0** (2025-01-20): Mobile portrait detection, distributed analysis
- **v4.15.0** (2025-01-20): Balanced 50/50 face/body weighting
- **v4.13.0** (2025-09-08): Unified processing methods, code simplification

### Key Improvements
- ✅ YOLOv8 mandatory for optimal accuracy
- ✅ Cross-platform compatibility (Windows/Linux/macOS)
- ✅ Security hardening and input validation
- ✅ Comprehensive test suite with 35+ tests
- ✅ Automated CI/CD with coverage reporting
- ✅ Enhanced error handling and user feedback

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes with tests
4. Run test suite: `make test`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Create Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pre-commit pytest pytest-cov

# Run quality checks
make lint
make format
make test
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Repository**: https://github.com/vasil-derimanov/svod
- **Issues**: https://github.com/vasil-derimanov/svod/issues
- **Wiki**: https://github.com/vasil-derimanov/svod/wiki

## 🙏 Acknowledgments

- OpenCV community for computer vision libraries
- YOLO authors for object detection framework
- Intel OpenVINO team for optimization tools
- PyTorch community for machine learning support
- Ultralytics team for YOLOv8 implementation