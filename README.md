# Smart Video Orientation Detector (SVOD) v4.6.0

🎥 **Intelligent video orientation detection using advanced computer vision and machine learning techniques**

## 📋 Overview

SVOD automatically detects and analyzes video orientation using multiple detection methods:
- **Face Detection** - Primary orientation indicator using DNN face detector
- **Body Detection** - Secondary analysis using YOLO v4 object detection  
- **Enhanced Detection** - Optional MobileNet models for improved accuracy
- **Facial Landmarks** - Precise orientation analysis using LBF landmark detection

## 🚀 Key Features

- ✅ **Single Video Analysis** - Process individual videos with detailed orientation reports
- ✅ **Batch Processing** - Process entire folders recursively with comprehensive reports
- ✅ **Cross-Platform** - Works on Windows, macOS, and Linux
- ✅ **Real-time Display** - Optional live preview during analysis
- ✅ **Flexible Output** - Save annotated videos and detailed CSV/JSON reports
- ✅ **Performance Optimized** - Time-limited analysis for large video sets
- ✅ **Validation Mode** - Compare results against reference orientation data

## 📦 Dependencies

Install all required packages using pip:

```bash
# Core computer vision libraries
pip install opencv-python
pip install opencv-contrib-python
pip install numpy

# Machine learning frameworks
pip install onnx
pip install torch
pip install torchvision

# Intel OpenVINO (optional for enhanced detection)
pip install openvino
pip install openvino-dev
```

**System Requirements:**
- Python 3.8+ (recommended: Python 3.11+)
- 4GB+ RAM for optimal performance
- GPU support optional but recommended for large batch processing

## 🔧 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/vasil-derimanov/svod.git
cd svod
```

2. **Install dependencies:**
```bash
pip install opencv-python opencv-contrib-python numpy onnx torch torchvision openvino openvino-dev
```

3. **Download required model files:**
The script will automatically download all required model files on first run:
- YOLO v4 configuration and weights
- DNN face detector model
- Facial landmark detection model  
- MobileNet models (optional enhancement)

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

## 🙏 Acknowledgments

- OpenCV community for computer vision libraries
- YOLO authors for object detection framework
- Intel OpenVINO team for optimization tools
- PyTorch community for machine learning support