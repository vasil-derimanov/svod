# SVOD - Smart Video Orientation Detector

## Project Overview
SVOD detects video orientation (CORRECT/INCORRECT/UNCERTAIN) using AI-powered computer vision. The core is a ~5,700-line monolithic Python module (`video_orientation_detector.py`) that combines multiple detection models through a weighted voting system.

**Current Version:** 4.27.0 - Developer Experience Release (April 2026)
**Previous Milestones:**
- v4.26.0 - Performance Optimization Release (adaptive cascade, early exit)
- v4.25.0 - YuNet & Direction Accuracy Release (100% direction accuracy)
- v4.24.0 - Housekeeping release
- v4.23.0 - 100% orientation accuracy on reference dataset

## Architecture & Detection Pipeline

### Multi-Model Ensemble Approach
SVOD uses a **weighted voting system** across multiple detection methods:
1. **YOLOv11 Person Detection** (PRIMARY, REQUIRED) - ONNX Runtime models (`yolo11n.onnx` + `yolo11n-pose.onnx`) with PyTorch fallback
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

### Performance Optimizations (v4.26.0+)
- **Adaptive model cascade**: Skips Haar cascade when YuNet detects faces; skips tier-3 models (MobileNet, Hough, aspect) when YOLO+face detectors agree
- **Early exit**: Stops frame processing when verdict is decisive (batch: 50% coverage/90% ratio; single: 80% coverage/95% ratio)
- **Reduced probe**: Rotation direction probe samples 3 frames instead of 5
- Stats tracked: `cascade_skipped_haar`, `cascade_skipped_tier3`
- **Critical**: MediaPipe Pose must NOT be skipped even when YOLO detects bodies — provides independent orientation angles needed for edge cases (P7061239.mp4)

### v4.27.0 Enhancements
- **ONNX Runtime inference**: YOLO models exported to ONNX for faster CPU inference via `onnxruntime`; PyTorch fallback when ONNX unavailable
- **Parallel model inference**: `ThreadPoolExecutor(max_workers=4)` runs YuNet/YOLO/Pose/FaceMesh concurrently per frame
- **Frame downscaling**: `_max_detect_dim = 640` limits detection input size for speed
- **Smart pre-scan**: `--quick-scan` probes for human-rich segments before full analysis
- **3-tier direction voting ratio**: Stricter thresholds for small sample sizes (`<8 votes: 3.0`, `8-14: 2.0`, `≥15: 1.5`)
- **CLI refactoring**: CLI code extracted to `svod_cli.py`; statistics mixin to `svod_statistics.py`; data models to `svod_models.py`
- **New CLI flags**: `--json` (JSON output), `--auto-correct` (ffmpeg rotation), `--export-onnx` (model conversion), `--frame-size`, `--quick-scan`
- **Progress callback**: `process_video_unified()` accepts optional `progress_callback` for integration

### Direction Detection Pipeline
Direction is determined through a priority chain in `resolve_rotation_direction()`:
1. **Face position direction** (face on left/right side of frame, 1.1x threshold)
2. **Preferred direction** (from weighted strength voting or count majority with sample-size-aware ratio)
3. **Rotation direction counts** (3-tier sample-size-aware ratio: `<8 votes: 3.0`, `8-14: 2.0`, `≥15: 1.5`)
4. **Physical rotation probe** (`_probe_rotation_direction()`)
5. **Default: counterclockwise** (statistically most common for phone-recorded sideways video)

### Landscape-Portrait Content Detection
**Critical pattern**: Landscape videos may contain portrait content (e.g., `P2170127.mp4`). Detection relies on:
- Detection box aspect ratios (wide boxes in landscape = portrait content)
- Face positioning (left/right sides suggest rotation)
- Strong bias values (15.0+) for decisive verdicts

## Development Workflows

### Testing Philosophy: Two Distinct Ecosystems
**`tests/` directory** - Automated pytest suite (19 test files, 209 tests):
- Unit tests: `test_basic.py`, `test_core_detection.py`, `test_face_detection.py`
- Integration tests: `test_integration.py`, `test_model_integration.py`
- Feature tests: `test_new_features.py`, `test_performance_optimizations.py`
- Run with: `pytest tests/` or `make test`
- Coverage requirement: 15% minimum (see `pyproject.toml`), current: ~33%

**`testing/` directory** - Manual real-video testing scripts:
- `standard_single_test.py` - Individual video testing
- `standard_batch_test.py` - Batch folder validation with reference data comparison
- `standard_performance_test.py` - Performance benchmarking
- **NEVER create new test scripts** - always use these 3 standard scripts

### Reference Validation System
**`reference_orientations.csv`** - Ground truth dataset (37 videos with known orientations):
- Format: `filename,expected_orientation,confidence,notes`
- **NEVER modify reference_orientations.csv** - it is the ground truth; fix detection code instead
- Current accuracy: **100% orientation (37/37), 100% direction (37/37)**

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
- `yolo11n.onnx` - YOLOv11 nano detection model (ONNX Runtime, exported via `--export-onnx`)
- `yolo11n-pose.onnx` - YOLOv11 nano pose model (ONNX Runtime)
- `yolo11n.pt`, `yolo11n-pose.pt` - PyTorch fallback models
- `face_detection_yunet_2023mar.onnx` - YuNet face detector
- `coco.names` - YOLO class names
- `mobilenet-v2.xml`, `mobilenet-v2.bin` - MobileNet OpenVINO models (optional)

### Pinned Package Versions (requirements.txt)
- `opencv-contrib-python==4.11.0.86` (contrib required by mediapipe; never install opencv-python alongside)
- `numpy==1.26.4` (v2.0+ incompatible)
- `ultralytics>=8.3.0` (YOLOv11)
- `torch==2.8.0`, `torchvision==0.23.0`
- `openvino==2024.6.0`
- `onnxruntime>=1.17.0` (fast CPU inference for YOLO ONNX models)
- `mediapipe>=0.10.0`

## Key Files & Directories

- **`video_orientation_detector.py`** - Core detection module (~5,700 lines)
- **`svod_cli.py`** - CLI argument parsing, validation, and entry points
- **`svod_statistics.py`** - `StatisticsMixin` for statistics collection and reporting
- **`svod_models.py`** - Data models and JSON serialization (`results_to_json`, `batch_results_to_json`)
- **`pyproject.toml`** - Project metadata, `svod` CLI entry point
- **`.flake8`** - Flake8 linting configuration (authoritative config file)
- **`reference_orientations.csv`** - Ground truth dataset (37 videos)
- **`inspect_rotation.py`** - Developer utility for debugging rotation direction
- **`performance_baselines/`** - Version performance benchmarks
- **`YOLOV10_UPGRADE.md`** - Historical migration documentation

## Common Pitfalls & Solutions

1. **OpenCV conflict**: Only install `opencv-contrib-python` — never `opencv-python` alongside it
2. **Stale __pycache__**: Always `Remove-Item -Recurse -Force __pycache__` before testing code changes
3. **Batch test encoding**: Always set `$env:PYTHONIOENCODING='utf-8'` before running
4. **NumPy v2.0**: Pin to `numpy==1.26.4`
5. **MediaPipe unavailable**: SVOD injects a lightweight stub; install `mediapipe` for full support
6. **YOLOv11 "not available" error**: Install ultralytics: `pip install "ultralytics>=8.3.0"`
7. **MobileNet download fails**: Expected on Apple Silicon — core detection still accurate
8. **"omz_downloader not found"**: Requires Python 3.11-3.12, install `openvino-dev`

## Real-World Testing

- **Good_Examples/**: Expected >95% CORRECT classification (currently 22/22 = 100%)
- **Bad_Examples/**: Expected INCORRECT classification (currently 15/15 = 100%)
- Each version has a baseline file in `performance_baselines/`
- v4.27.0 maintains 100% orientation + 100% direction accuracy

## Adding New Features

When extending detection capabilities:
1. Add new detection method to `OrientationDetector` class
2. Integrate into `determine_frame_orientation()` voting logic
3. Add statistics tracking to `self.stats` dictionary
4. Update `print_statistics()` to display new metrics
5. Add unit tests to `tests/test_*.py` AND manual tests with `testing/standard_*_test.py`
6. Update `README.md` usage examples

## Documentation Standards

- **README.md**: User-facing documentation with CLI examples
- **Code comments**: Inline explanations for complex detection logic
- **Docstrings**: All public methods have detailed docstrings with Args/Returns
- **YOLOV10_UPGRADE.md**: Historical migration documentation (no longer actively maintained)
