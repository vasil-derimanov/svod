# Smart Video Orientation Detector (SVOD) v4.8.0

🎥 **Intelligent video orientation detection with enhanced DNN validation and automated cleanup**

## 📋 Overview

SVOD automatically detects and analyzes video orientation using multiple detection methods:
- **Face Detection** - Primary orientation indicator using DNN face detector
- **Body Detection** - Secondary analysis using YOLO v4 object detection  
- **Enhanced Detection** - MobileNet models for improved accuracy
- **Facial Landmarks** - Precise orientation analysis using LBF landmark detection

## 🚀 Key Features

- ✅ **Single Video Analysis** - Process individual videos with detailed orientation reports
- ✅ **Batch Processing** - Process entire folders recursively with comprehensive reports
- ✅ **Cross-Platform** - Works on Windows, macOS, and Linux
- ✅ **Enhanced DNN Validation** - Robust error handling and model validation (v4.8.0)
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

All model files (YOLO, DNN, MobileNet, etc.) are downloaded automatically on first run.

**System Requirements:**
- Python 3.8+ (recommended: Python 3.11+)
- 4GB+ RAM for optimal performance
- Internet connection for initial model downloads
- No optional components - all models are mandatory for operation

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

## 📊 Output Formats

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

## 📝 Version History

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

## �🙏 Acknowledgments

- OpenCV community for computer vision libraries
- YOLO authors for object detection framework
- Intel OpenVINO team for optimization tools
- PyTorch community for machine learning support