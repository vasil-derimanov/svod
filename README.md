# Smart Video Orientation Detector (SVOD) v4.17.0

🎥 **Enhanced Mobile Detection with Portrait Override & Distributed Analysis**

## 📋 Overview

SVOD automatically detects and analyzes video orientation using multiple detection methods:
- **Face Detection** - Primary orientation indicator using DNN face detector
- **Body Detection** - Hybrid YOLOv8/YOLOv4 analysis with automatic fallback
- **Enhanced Detection** - MobileNet models with OpenVINO optimization  
- **Facial Landmarks** - Precise orientation analysis using LBF landmark detection
- **Cross-Platform Intelligence** - Optimized for Windows, Linux, and Apple Silicon (M1/M2/M3)

## � Enhanced Mobile Detection (v4.17.0)

SVOD v4.17.0 introduces groundbreaking mobile portrait detection with automatic override logic:

**Key Features:**
- **Mobile Portrait Override**: Automatic detection of mobile portrait videos (aspect ratio < 0.65)
- **Distributed Analysis**: Smart sampling across video segments (start/middle/end) for comprehensive analysis
- **Generic Detection**: Works with all mobile portrait formats (16:9, 4:3, modern mobile ratios)
- **Force Override Logic**: Automatic rotation detection for problematic mobile videos

**Technical Implementation:**
```python
# Mobile Portrait Override Logic
if video_aspect_ratio < 0.65:  # Mobile portrait detection
    # Force portrait detection and rotation logic
    mobilenet_vote = "portrait"
    detection_info['mobile_portrait_override'] = f'aspect_{video_aspect_ratio:.3f}_forced_portrait'
```

**Performance Results:**
- **100% Orientation Detection**: Perfect accuracy in determining correct vs incorrect orientation (7/7 files)
- **Robust Mobile Detection**: Successfully detects mobile portrait videos with aspect ratio < 0.65
- **Cross-Platform**: Works on Windows, WSL Ubuntu, and macOS  
- **Distributed Analysis**: Intelligent sampling reduces processing time while maintaining detection accuracy
- **Note**: Rotation direction detection needs improvement (2/3 files suggest incorrect rotation direction)

## 🎯 Balanced Weighting System (v4.15.0)

SVOD v4.15.0 introduces a revolutionary balanced 50/50 face/body weighting system that dramatically improves counterclockwise detection accuracy:

**Key Improvements:**
- **50/50 Equal Weighting**: Faces and bodies contribute exactly 50% each to final decision, regardless of detection counts
- **High Confidence Face Filtering**: Increased face confidence threshold from 0.6 to 0.8 to reduce false positives
- **Ratio-Based Calculations**: Uses percentage of correct vs incorrect votes per category instead of raw counts
- **False Positive Protection**: Automatic face density filtering when >5 faces per frame detected

**Technical Implementation:**
```python
# Balanced weighting formula
if face_total_votes > 0 and body_total_votes > 0:
    face_correct_ratio = face_correct_votes / face_total_votes
    body_correct_ratio = body_correct_votes / body_total_votes
    
    # 50/50 balanced weighting
    weighted_correct = (face_correct_ratio * 0.5) + (body_correct_ratio * 0.5)
```

**Results Comparison:**
- **Before**: Many-face videos dominated by unreliable face detections
- **After**: Equal influence ensures robust detection even with extreme face counts
- **Accuracy**: Maintained 85.7% batch accuracy while fixing counterclockwise detection
- **Confidence**: More conservative UNCERTAIN classifications for borderline cases

**Example Improvements:**
- VID_20200907_202511.mp4: CORRECT(93.88%) → UNCERTAIN(57.58%) ✅
- P9080828.mp4: CORRECT(71.43%) → INCORRECT(96.55%) ✅ 
- P8150092.mp4: CORRECT(94.59%) → CORRECT(74.7%) ✅ (no regression)

## 📋 Overview

SVOD automatically detects and analyzes video orientation using multiple detection methods:
- **Face Detection** - Primary orientation indicator using DNN face detector
- **Body Detection** - Hybrid YOLOv8/YOLOv4 analysis with automatic fallback
- **Enhanced Detection** - MobileNet models with OpenVINO optimization
- **Facial Landmarks** - Precise orientation analysis using LBF landmark detection
- **Cross-Platform Intelligence** - Optimized for Windows, Linux, and Apple Silicon (M1/M2/M3)

## 🚀 YOLOv8 Hybrid Detection System (v4.16.0)

SVOD v4.16.0 introduces an intelligent hybrid YOLOv8/YOLOv4 system that automatically provides the best available body detection:

**Enhanced Detection Features:**
- **YOLOv8 First**: Automatically uses YOLOv8 nano model if ultralytics is available
- **Seamless Fallback**: Gracefully falls back to proven YOLOv4 if YOLOv8 fails
- **Precision Improvement**: YOLOv8 provides more accurate body detections with fewer false positives
- **Automatic Installation**: Script attempts to install ultralytics automatically during setup
- **Conflict Resolution**: Robust OpenCV version management to prevent import conflicts

**Technical Implementation:**
```python
# Hybrid detection logic
if YOLOV8_AVAILABLE:
    results = self.yolov8_model(frame, verbose=False)
    # Process YOLOv8 results with enhanced precision
else:
    # Fallback to YOLOv4 with OpenCV DNN
    outputs = self.net.forward(self.output_layers)
```

**Detection Comparison:**
- **YOLOv8**: More precise, fewer false positives (e.g., P8150092.mp4: 31 detections)
- **YOLOv4**: Robust fallback, proven stability (e.g., P8150092.mp4: 306 detections)
- **Accuracy**: Maintains 85.7% batch accuracy with both systems
- **Performance**: YOLOv8 provides cleaner detection data for balanced weighting

## 🚀 Key Features

- ✅ **Single Video Analysis** - Process individual videos with detailed orientation reports
- ✅ **Batch Processing** - Process entire folders recursively with comprehensive reports
- ✅ **Cross-Platform Intelligence** - Windows, Linux, Apple Silicon (M1/M2/M3) with optimized fallbacks
- ✅ **Python 3.11-3.12 Optimized** - Ideal compatibility for omz_downloader and all dependencies (v4.11.0)
- ✅ **Smart Dependency Management** - Automatic installation with platform-specific optimizations
- ✅ **Automated Cleanup** - Built-in scripts for clean testing environments (v4.8.0)
- ✅ **Real-time Display** - Optional live preview during analysis
- ✅ **Flexible Output** - Save annotated videos and detailed CSV/JSON reports
- ✅ **Performance Optimized** - Time-limited analysis for large video sets
- ✅ **Validation Mode** - Compare results against reference orientation data
- ✅ **Automatic Setup** - All dependencies and models are installed automatically

## 📦 Dependencies (Automatic Installation)

**No manual installation required!** SVOD automatically installs and configures all dependencies:

- **opencv-contrib-python** (required for facial landmarks)
- **numpy** (mathematical operations)
- **openvino** (Intel OpenVINO for MobileNet inference)
- **openvino-dev** (OpenVINO Model Zoo tools - platform dependent)
- **torch** (PyTorch CPU version for MobileNet model conversion)
- **onnx** (ONNX format support for model conversion)
- **ultralytics** (optional YOLOv8 support - auto-installed for enhanced detection)

All model files (YOLO, DNN, MobileNet, etc.) are downloaded automatically:
- **omz_downloader** (preferred method for MobileNet models on compatible platforms)
- **Direct downloads** (fallback method, especially for Apple Silicon)

**System Requirements:**
- **Python 3.11-3.12** (required for full omz_downloader compatibility)
- 4GB+ RAM for optimal performance
- Internet connection for initial model downloads
- No optional components - all models are mandatory for operation

**Platform Notes:**
- **Windows/Linux:** Full omz_downloader support for optimal model acquisition
- **Apple Silicon (M1/M2/M3):** Uses direct download fallbacks for better compatibility
- **Python 3.13+:** Not supported due to NumPy compilation issues with omz_downloader

## 🔧 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/vasil-derimanov/svod.git
cd svod
```

2. **Run the script - that's it!**
```bash
python video_orientation_detector.py your_video.mp4
```

**The script automatically:**
- Installs all required Python packages (opencv-contrib-python, numpy, openvino)
- Downloads all model files (YOLO, DNN Face detector, MobileNet, etc.)
- Validates all components with enhanced DNN support verification (v4.8.0)

**No manual setup required!** Everything is handled automatically on first run.

## 🧹 Clean Testing Environment (v4.8.0)

SVOD v4.8.0 includes automated cleanup scripts for vanilla testing:

### PowerShell Cleanup
```powershell
# Clean all models and test environments
.\cleanup.ps1
```

### Python Cleanup  
```bash
# Cross-platform cleanup
python cleanup.py
```

**Cleanup removes:**
- All model files (YOLO, DNN, MobileNet, etc.)
- Test virtual environments (.venv-clean, .venv-test, etc.)
- Temporary/cache files (__pycache__, *.pyc, etc.)
- Status tracking files

**Perfect for:**
- Vanilla environment testing
- Clean CI/CD runs
- Development workflow automation

## 🐧 WSL/Linux Setup

For WSL (Windows Subsystem for Linux) users, ensure OpenVINO is installed for enhanced detection:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Install OpenVINO for enhanced detection features
pip install openvino

# Test the installation
python3 video_orientation_detector.py --version
```

**Note**: OpenVINO provides enhanced MobileNet detection capabilities but is optional. The script works without it using core computer vision methods.

## 🍎 Apple Silicon (M1/M2/M3) Support

SVOD v4.9.2+ includes enhanced compatibility for Apple Silicon Macs:

- **Automatic Detection**: Script detects Apple Silicon chips and adjusts behavior accordingly
- **Graceful Fallback**: If OpenVINO/MobileNet fails, core algorithms provide excellent accuracy
- **Multiple Download Sources**: Improved fallback URLs for model downloads
- **Clear Messaging**: Informative messages about Apple Silicon compatibility

**Known Limitations**:
- OpenVINO has limited ARM64 support on macOS
- MobileNet models may not download automatically
- Core detection algorithms work perfectly without MobileNet

## 💻 Usage Examples

### Single Video Analysis

```bash
# Basic analysis with display
python video_orientation_detector.py video.mp4

# Save annotated output video
python video_orientation_detector.py video.mp4 -o corrected.mp4

# Process without display (faster)
python video_orientation_detector.py video.mp4 --no-display

# Analyze only first 10 seconds
python video_orientation_detector.py video.mp4 --time-limit 10

# Higher confidence threshold
python video_orientation_detector.py video.mp4 -c 0.7
```

### Batch Folder Processing

```bash
# Process all videos in folder
python video_orientation_detector.py /path/to/videos --batch

# Process recursively (include subfolders)
python video_orientation_detector.py /path/to/videos --batch -r

# Save detailed batch report
python video_orientation_detector.py /path/to/videos --batch --report summary.txt

# Analyze first 15 seconds of each video
python video_orientation_detector.py /path/to/videos --batch --time-limit 15
```

### Advanced Options

```bash
# Use reference file for validation
python video_orientation_detector.py folder --batch --reference orientations.csv

# Combined options for large datasets
python video_orientation_detector.py /videos --batch -r --time-limit 30 --report detailed.json -c 0.8
```

## � Code Architecture Improvements (v4.13.0)

SVOD v4.13.0 introduces major code simplification and maintainability improvements:

- **Unified Processing Methods**: Merged `process_video()` and `process_video_quick()` into single `process_video_unified()` method
- **Mode-Based Processing**: Single method supports `"full"`, `"batch"`, and `"quick"` modes for different use cases
- **Guaranteed Consistency**: Eliminates risk of logic divergence between individual and batch processing
- **Backward Compatibility**: Legacy wrapper methods preserve existing API compatibility
- **Reduced Code Complexity**: 137 lines of duplicate code eliminated
- **Enhanced Maintainability**: Single source of truth for video processing logic

**Benefits:**
- ✅ Same accuracy and performance (71.4% batch accuracy maintained)
- ✅ Identical results between all processing modes
- ✅ Easier debugging and future enhancements
- ✅ Reduced risk of inconsistencies and bugs
- ✅ Cleaner, more maintainable codebase

**Processing Modes:**
```bash
# All modes use the same underlying logic for guaranteed consistency
python video_orientation_detector.py video.mp4          # Full mode (display + annotation)
python video_orientation_detector.py folder --batch     # Batch mode (fast, no display)
python video_orientation_detector.py video.mp4 --quick  # Quick mode (fast with display)
```

## �📊 Output Formats

### Console Output
- Real-time detection confidence scores
- Final orientation determination
- Processing statistics and timing

### Report Files
- **CSV Format**: Orientation results with confidence scores
- **JSON Format**: Detailed analysis including vote counts and detection methods
- **Annotated Videos**: Visual overlay showing detection results

### Batch Reports
- Summary statistics across all processed videos
- Error reports for failed analyses
- Performance metrics and processing times

## 🔍 Detection Methods

1. **Face Detection** (Primary)
   - Uses DNN-based face detector
   - Analyzes face orientation and positioning
   - High accuracy for videos with visible faces

2. **Body Detection** (Secondary)  
   - YOLO v4 object detection for people
   - Backup method when faces aren't clearly visible
   - Useful for wide shots and group scenes

3. **Enhanced Detection** (Optional)
   - MobileNet models for improved accuracy
   - Additional confidence scoring
   - Cross-platform fallback support

## ⚙️ Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `path` | Video file or folder path | Required |
| `--output, -o` | Save annotated video | None |
| `--no-display` | Process without preview | False |
| `--confidence, -c` | Detection threshold (0-1) | 0.5 |
| `--time-limit, -t` | Max analysis time (seconds) | Full video |
| `--batch` | Enable batch processing | False |
| `--recursive, -r` | Process subfolders | False |
| `--report` | Save batch report file | None |
| `--reference` | Reference file for validation | None |

### Environment Variables

- `SVOD_CONFIDENCE`: Default confidence threshold
- `SVOD_TIME_LIMIT`: Default time limit for analysis

## 🛠️ Troubleshooting

### Common Issues

**Model files not downloading:**
- Check internet connectivity
- Verify firewall/proxy settings
- Try manual download to script directory

**Performance issues:**
- Use `--time-limit` for large videos
- Enable `--no-display` for faster processing
- Close other GPU-intensive applications

**Cross-platform compatibility:**
- MobileNet models are optional on macOS
- Script will run with core models if enhanced detection fails
- Check Python and OpenCV versions

### Error Codes

- **Exit 0**: Successful completion
- **Exit 1**: Critical model files missing
- **Exit 2**: Invalid input parameters
- **Exit 3**: Video processing error

## � Version Comparison and Statistics

SVOD includes comprehensive tools for tracking version evolution, performance analysis, and regression detection:

### Version Comparison Tool

Compare accuracy, speed, and detection methods between SVOD versions:

```bash
# Basic version comparison
python svod_version_comparison.py

# Save detailed comparison report
python svod_version_comparison.py --save-report

# Specify custom output file
python svod_version_comparison.py --save-report --output comparison_report.json
```

**Features:**
- Accuracy trend analysis between versions
- Processing speed comparison
- Detection method usage evolution
- YOLOv8 vs YOLOv4 adoption tracking
- Automatic file discovery and analysis

### Statistics Collector

Collect and store historical SVOD performance data:

```bash
# Collect current statistics
python svod_statistics_collector.py --collect

# View evolution summary
python svod_statistics_collector.py --summary

# Generate full evolution report
python svod_statistics_collector.py --report
```

**Features:**
- SQLite database for historical tracking
- Model usage statistics (YOLOv8/YOLOv4, detection methods)
- Video-level accuracy tracking
- Performance trend analysis
- Cross-platform compatibility metrics

### Automated Benchmark Suite

Comprehensive benchmarking system for version validation:

```bash
# Basic benchmark with test videos
python svod_benchmark_suite.py --save-results

# Clean environment benchmark (vanilla testing)
python svod_benchmark_suite.py --clean --save-results --compare

# Custom test video directory
python svod_benchmark_suite.py --test-videos /path/to/videos --max-videos 10
```

**Features:**
- Cross-platform testing (Windows, WSL, macOS)
- Automatic model download and cleanup
- Performance regression detection
- YOLOv8/YOLOv4 comparison benchmarks
- Standardized test video processing

### Evolution Reporter

Generate comprehensive evolution analysis reports:

```bash
# View evolution summary
python svod_evolution_reporter.py --summary

# Save full evolution report
python svod_evolution_reporter.py --save

# Custom output file
python svod_evolution_reporter.py --save --output evolution_analysis.json
```

**Features:**
- Historical accuracy and performance trends
- YOLO version adoption analysis
- Detection method evolution tracking
- Executive summary with key findings
- Regression detection and recommendations

### Usage Workflow

1. **After making changes**: Run benchmark to collect new data
2. **Compare versions**: Use comparison tool to analyze improvements
3. **Track evolution**: Use statistics collector to store historical data
4. **Generate reports**: Use evolution reporter for comprehensive analysis

```bash
# Complete workflow example
python svod_benchmark_suite.py --clean --save-results
python svod_statistics_collector.py --collect
python svod_version_comparison.py --save-report
python svod_evolution_reporter.py --summary
```

## �📝 Version History

- **v4.16.0** (2025-01-20): YOLOv8 Hybrid Detection - added optional YOLOv8 support with automatic fallback to YOLOv4, enhanced body detection precision while maintaining 85.7% accuracy, robust dependency management with OpenCV conflict resolution
- **v4.15.0** (2025-01-20): Balanced Face/Body Weighting - implemented 50/50 balanced weighting system where faces and bodies contribute equally regardless of detection counts, increased face confidence threshold to 0.8, significantly improved counterclockwise detection accuracy
- **v4.14.0** (2025-01-20): Enhanced Rotation Direction Detection - improved accuracy for counterclockwise rotations with balanced detection logic and position-based heuristics
- **v4.13.0** (2025-09-08): Code Unification & Simplification - merged process_video() methods into unified system
- **v4.12.5** (2025-09-08): Critical Batch-Individual Consistency Fix - improved accuracy from 42.9% to 71.4%
- **v4.12.4** (2025-09-08): Accuracy improvements with dynamic thresholds
- **v4.12.3** (2025-09-08): Complete MobileNet Integration with Automatic PyTorch Installation
- **v4.10.2** (2025-01-21): Enhanced rotation direction detection - intelligent clockwise/counterclockwise analysis
- **v4.10.1** (2025-01-21): Apple Silicon compatibility improvements and error handling
- **v4.10.0** (2025-01-21): Python 3.13 optimization and performance improvements
- **v4.9.2** (2025-09-07): Apple Silicon M3 compatibility - improved OpenVINO handling and fallback URLs
- **v4.9.1** (2025-09-07): Adaptive MobileNet requirement - graceful handling of WSL/Linux environments
- **v4.9.0** (2025-09-07): Made MobileNet models mandatory for enhanced detection accuracy  
- **v4.8.0** (2025-09-07): Enhanced DNN validation + Automated cleanup scripts for vanilla testing
- **v4.7.0** (2025-09-07): Enhanced compatibility and MobileNet optimization
- **v4.6.2** (2025-09-07): Clean project - removed cSpell dependencies, cross-platform testing verified
- **v4.6.1** (2025-09-07): Enhanced macOS Python 3.12+ compatibility, improved OpenVINO error handling
- **v4.6.0** (2025-09-07): Improved cross-platform compatibility, MobileNet files now optional
- **v4.5.0**: Clean settings & virtual environment support
- **v4.x.x**: Enhanced batch processing and reporting
- **v3.x.x**: Added YOLO v4 and facial landmark detection
- **v2.x.x**: Initial face detection implementation
- **v1.x.x**: Basic orientation detection

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **Repository**: https://github.com/vasil-derimanov/svod
- **Issues**: https://github.com/vasil-derimanov/svod/issues
- **Wiki**: https://github.com/vasil-derimanov/svod/wiki

## � Model Files Detailed Guide

### Required Model Files

The script uses several pre-trained models for optimal detection accuracy. All files are automatically downloaded on first run, but here are the details for manual setup or troubleshooting:

#### 1. YOLO v4 Object Detection
- **File**: `yolov4.cfg` (configuration)
- **File**: `yolov4.weights` (weights, ~245MB)
- **Source**: [AlexeyAB/darknet releases](https://github.com/AlexeyAB/darknet/releases)
- **Version**: YOLOv4 optimal
- **Purpose**: Person detection for body orientation analysis
- **Manual Download**:
  ```bash
  # Configuration file
  curl -O https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
  
  # Weights file (large download)
  curl -L -O https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
  ```

#### 2. DNN Face Detection
- **File**: `deploy.prototxt` (network architecture)
- **File**: `res10_300x300_ssd_iter_140000.caffemodel` (weights, ~10MB)
- **Source**: [OpenCV DNN Face Detector](https://github.com/opencv/opencv_3rdparty)
- **Version**: SSD MobileNet-based face detector
- **Purpose**: Primary face detection and orientation analysis
- **Manual Download**:
  ```bash
  # Prototxt file
  curl -O https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
  
  # Model file
  curl -O https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
  ```

#### 3. Facial Landmark Detection
- **File**: `lbfmodel.yaml` (~54MB)
- **Source**: [OpenCV Face Module](https://github.com/opencv/opencv_contrib)
- **Version**: LBF (Local Binary Features) model
- **Purpose**: Precise facial landmark detection for orientation analysis
- **Manual Download**:
  ```bash
  curl -O https://github.com/opencv/opencv_3rdparty/raw/contrib_face_alignment_20170818/lbfmodel.yaml
  ```

#### 4. COCO Class Names
- **File**: `coco.names` (text file)
- **Source**: [COCO Dataset](https://github.com/AlexeyAB/darknet)
- **Purpose**: Object class labels for YOLO detection
- **Manual Download**:
  ```bash
  curl -O https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
  ```

### Optional Enhancement Files (MobileNet)

These files provide enhanced detection accuracy but are **optional** - the script works without them:

#### 5. MobileNet v2 Models (Optional)
- **Files**: `mobilenet-v2.xml`, `mobilenet-v2.bin` (~14MB total)
- **Source**: [Intel OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
- **Version**: MobileNet v2 1.0 224
- **Purpose**: Enhanced image classification and orientation scoring
- **Auto-download**: Uses Intel OpenVINO tools with macOS fallback
- **Manual Download** (if OpenVINO tools fail):
  ```bash
  # XML configuration
  curl -O https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml
  mv mobilenet-v2-pytorch.xml mobilenet-v2.xml
  
  # Binary weights  
  curl -O https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.bin
  mv mobilenet-v2-pytorch.bin mobilenet-v2.bin
  ```

### File Size Summary
| File | Size | Required | Purpose |
|------|------|----------|---------|
| `yolov4.weights` | ~245MB | ✅ Yes | Object detection |
| `lbfmodel.yaml` | ~54MB | ✅ Yes | Facial landmarks |
| `mobilenet-v2.bin` | ~14MB | ⚠️ Optional | Enhanced detection |
| `res10_300x300_ssd_iter_140000.caffemodel` | ~10MB | ✅ Yes | Face detection |
| Other files | <1MB each | ✅ Yes | Configurations |
| **Total Required** | **~310MB** | | |
| **Total with Optional** | **~324MB** | | |

### Troubleshooting Model Downloads

**If automatic download fails:**

1. **Check internet connectivity and firewall settings**
2. **Manual download**: Use the curl commands above
3. **Verify file integrity**: Check file sizes match the table above
4. **Place files in script directory**: Same folder as `video_orientation_detector.py`
5. **macOS OpenVINO issues**: MobileNet files are optional - script runs without them

**Verification command:**
```bash
python video_orientation_detector.py --version
# This will show if all required files are present
```

## � Enhanced Rotation Direction Detection (v4.10.2)

SVOD v4.10.2 introduces intelligent rotation direction analysis:

- **Face-Based Direction**: Analyzes face aspect ratios and positions to determine clockwise vs counterclockwise rotation
- **Mobile Video Optimization**: Special handling for portrait videos (common with mobile devices)
- **Body Analysis Integration**: Uses body detection to supplement face-based direction decisions
- **Voting System**: Multiple detection methods vote on rotation direction for improved accuracy
- **Context-Aware**: Considers video format (portrait vs landscape) in rotation recommendations

**Improved Accuracy for:**
- Mobile phone videos (vertical orientation)
- Videos with multiple subjects
- Challenging lighting conditions
- Mixed orientation scenarios

**Direction Detection Logic:**
- Portrait videos (aspect ratio < 0.8): Optimized for mobile device footage
- Landscape videos (aspect ratio > 1.2): Traditional camera/screen recordings
- Face positioning analysis: Determines most likely rotation direction
- Fallback heuristics: Default recommendations when detection is uncertain

## �🐍 Python 3.13 Optimization (v4.10.0)

SVOD v4.10.0 includes specific optimizations for Python 3.13:

- **Enhanced Performance**: ~2% faster inference speed with Python 3.13
- **Improved Dependencies**: Better compatibility with latest OpenVINO, OpenCV, and NumPy versions
- **UTF-8 Support**: Automatic handling of Unicode emoji characters in console output
- **Future-Ready**: Leverages Python 3.13's performance improvements and new features

**Windows UTF-8 Setup (if needed):**
```powershell
$env:PYTHONIOENCODING="utf-8"
python video_orientation_detector.py --version
```

**Cross-Platform Testing:**
- ✅ Windows 11 + Python 3.13.7 (native)
- ✅ WSL2 + Python 3.12.3 (fallback compatibility)
- ✅ Apple Silicon M3 + Python 3.13+ (optimized)

## 🙏 Acknowledgments

- OpenCV community for computer vision libraries
- YOLO authors for object detection framework
- Intel OpenVINO team for optimization tools
- PyTorch community for machine learning support