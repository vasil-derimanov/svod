# SVOD - Smart Video Orientation Detector

## Project Overview
SVOD detects video orientation (CORRECT/INCORRECT/UNCERTAIN) using AI-powered computer vision. The core is a ~6,200-line monolithic Python module (`video_orientation_detector.py`) that combines multiple detection models through a weighted voting system.

**Current Version:** 4.25.0 - YuNet & Direction Accuracy Release (March 2026)
**Previous Milestones:**
- v4.24.0 - Housekeeping release
- v4.23.0 - 100% orientation accuracy on reference dataset

## Architecture & Detection Pipeline

### Multi-Model Ensemble Approach
SVOD uses a **weighted voting system** across multiple detection methods:
1. **YOLOv11 Person Detection** (PRIMARY, REQUIRED) - `yolo11n.pt` + `yolo11n-pose.pt` models via ultralytics
2. **YuNet Face Detection** - OpenCV FaceDetectorYN with ONNX model (`face_detection_yunet_2023mar.onnx`)
3. **MobileNet Classification** - OpenVINO 2024.6.0 models (`mobilenet-v2.xml/.bin`) - optional enhancement
4. **Haar Cascade Face Detection** - OpenCV fallback detector
5. **MediaPipe Face Mesh** - 468-landmark face mesh for advanced pose detection
6. **MediaPipe Pose** - Full-body pose estimation for human orientation

### Key Detection Logic (OrientationDetector class)
- `process_video_unified()` - Main processing loop analyzing frames
- `determine_frame_orientation()` - Per-frame analysis combining all models
- `calculate_final_verdict()` - Weighted voting across frame results
- `detect_rotation_direction()` - Per-frame CW/CCW detection using keypoint + face position evidence
- `resolve_rotation_direction()` - Aggregates per-frame direction votes with tie detection and probe fallback
- `_analyze_body_angle_from_keypoints()` - Nose-vs-hip analysis for rotation direction (weight 30.0)
- `_probe_rotation_direction()` - Physical frame rotation probe with result caching (`_probe_cache`)

### Direction Detection Pipeline
Direction is determined through a priority chain in `resolve_rotation_direction()`:
1. **Face position direction** (face on left/right side of frame)
2. **Preferred direction** (from weighted strength voting or count majority with 2:1 ratio)
3. **Rotation direction counts** (only if clear majority, not a tie)
4. **Physical rotation probe** (`_probe_rotation_direction()`)
5. **Default: counterclockwise** (statistically most common for phone-recorded sideways video)

### Landscape-Portrait Content Detection
**Critical pattern**: Landscape videos may contain portrait content (e.g., `P2170127.mp4`). Detection relies on:
- Detection box aspect ratios (wide boxes in landscape = portrait content)
- Face positioning (left/right sides suggest rotation)
- Strong bias values (15.0+) for decisive verdicts

## Development Workflows

### Testing Philosophy: Two Distinct Ecosystems
**`tests/` directory** - Automated pytest suite (17 test files):
- Unit tests: `test_basic.py`, `test_core_detection.py`, `test_face_detection.py`
- Integration tests: `test_integration.py`, `test_model_integration.py`
- Run with: `pytest tests/` or `make test`
- Coverage requirement: 15% minimum (see `pyproject.toml`)

**`testing/` directory** - Manual real-video testing scripts:
- `standard_single_test.py` - Individual video testing
- `standard_batch_test.py` - Batch folder validation with reference data comparison
- `standard_performance_test.py` - Performance benchmarking
- **NEVER create new test scripts** - always use these 3 standard scripts

### Reference Validation System
**`reference_orientations.csv`** - Ground truth dataset (18 videos with known orientations):
- Format: `filename,expected_orientation,confidence,notes`
- **NEVER modify reference_orientations.csv** - it is the ground truth; fix detection code instead
- Current accuracy: 100% orientation (37/37), 86.7% direction (13/15 Bad_Examples)

### Build & Development Commands
```bash
make install    # Install dependencies + pre-commit hooks
make format     # Black formatting (line length 100)
make lint       # Flake8 (config in .flake8 file)
make test       # Run pytest suite
make check      # format + lint + test
make clean      # Remove __pycache__, .coverage, etc.
make build      # Build distribution
```

**Required PowerShell environment for testing:**
```powershell
$env:PYTHONIOENCODING='utf-8'
$env:TF_CPP_MIN_LOG_LEVEL='3'
```

### Python Version Constraints
- **Required:** Python 3.11-3.12 (3.13+ not supported due to NumPy/omz_downloader issues)

## Project-Specific Conventions

### Security Hardening
- **Path validation:** Rejects directory traversal (`..`), null bytes, excessive lengths (>4096 chars)
- **Resource limits:** `--max-files 1000`, `--max-depth 10` for batch processing
- **Time limits:** Configurable analysis timeout to prevent resource exhaustion
- **Input sanitization:** All user paths through `os.path.abspath()` + suspicious pattern checks

### Enum-Based Orientation System
```python
class VideoOrientation(Enum):
    CORRECT = "CORRECT - Humans are upright"
    INCORRECT = "INCORRECT - Humans are sideways/rotated"
    UNCERTAIN = "UNCERTAIN - Cannot determine orientation"
```

## Critical Dependencies & Model Files

### Required Files (All auto-downloaded)
- `yolo11n.pt` - YOLOv11 nano detection model
- `yolo11n-pose.pt` - YOLOv11 nano pose model with keypoint detection
- `face_detection_yunet_2023mar.onnx` - YuNet face detector
- `coco.names` - YOLO class names
- `mobilenet-v2.xml`, `mobilenet-v2.bin` - MobileNet OpenVINO models (optional)

### Pinned Package Versions (requirements.txt)
- `opencv-contrib-python==4.11.0.86` (contrib required by mediapipe; never install opencv-python alongside)
- `numpy==1.26.4` (v2.0+ incompatible)
- `ultralytics>=8.3.0` (YOLOv11)
- `torch==2.8.0`, `torchvision==0.23.0`
- `openvino==2024.6.0`
- `mediapipe>=0.10.0`

## Key Files & Directories

- **`video_orientation_detector.py`** - Monolithic ~6,200-line core module
- **`pyproject.toml`** - Project metadata, `svod` CLI entry point
- **`.flake8`** - Flake8 linting configuration (authoritative config file)
- **`reference_orientations.csv`** - Ground truth dataset (18 videos)
- **`inspect_rotation.py`** - Developer utility for debugging rotation direction
- **`performance_baselines/`** - Version performance benchmarks
- **`YOLOV10_UPGRADE.md`** - Historical migration documentation

## Common Pitfalls

1. **OpenCV conflict**: Only install `opencv-contrib-python` — never `opencv-python` alongside it
2. **Stale __pycache__**: Always `Remove-Item -Recurse -Force __pycache__` before testing code changes
3. **Batch test encoding**: Always set `$env:PYTHONIOENCODING='utf-8'` before running
4. **NumPy v2.0**: Pin to `numpy==1.26.4`
5. **MediaPipe unavailable**: SVOD injects a lightweight stub; install `mediapipe` for full support
   - Each version has baseline file in `performance_baselines/`
   - v4.23.0 achieved 100% reference validation (documented milestone)

4. **Real-World Folders** - Manual testing with `Good_Examples/` and `Bad_Examples/`
   - Good_Examples: Expected >95% CORRECT classification
   - Bad_Examples: Expected mix of INCORRECT and UNCERTAIN (challenging cases)

## Common Pitfalls & Solutions

1. **YOLOv11 "not available" error**: Install ultralytics: `pip install "ultralytics>=8.3.0"`
2. **MobileNet download fails**: Expected on Apple Silicon - core detection still accurate
3. **NumPy v2.0 issues**: Pin to `numpy==1.26.4` (specified in requirements.txt)
4. **"omz_downloader not found"**: Requires Python 3.11-3.12, install `openvino-dev`
5. **UNCERTAIN verdicts**: Tune environment variables (see PowerShell commands above)
6. **MediaPipe unavailable**: SVOD injects a lightweight stub; install `mediapipe` for full pose support
7. **OpenCV conflict**: Only install `opencv-contrib-python` — never `opencv-python` alongside it

## Adding New Features

When extending detection capabilities:
1. Add new detection method to `OrientationDetector` class
2. Integrate into `determine_frame_orientation()` voting logic
3. Add statistics tracking to `self.stats` dictionary
4. Update `print_statistics()` to display new metrics
5. Add unit tests to `tests/test_*.py` AND manual tests with `testing/standard_*_test.py`
6. Update `README.md` usage examples and `YOLOV10_UPGRADE.md` if optimization-related

## Documentation Standards

- **README.md**: User-facing documentation with CLI examples
- **Code comments**: Inline explanations for complex detection logic
- **Docstrings**: All public methods have detailed docstrings with Args/Returns
- **Version docs**: `YOLOV10_UPGRADE.md` for architectural decisions and migration history
