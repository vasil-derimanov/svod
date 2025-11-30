# SVOD - Smart Video Orientation Detector

## Project Overview
SVOD detects video orientation (CORRECT/INCORRECT/UNCERTAIN) using AI-powered computer vision. The core is a 6,778-line monolithic Python module (`video_orientation_detector.py`) that combines multiple detection models through a weighted voting system.

**Current Version:** 4.24.0 - Housekeeping Release (November 2025)
**Previous Milestone:** v4.23.0 achieved 100% accuracy on reference dataset validation

## Architecture & Detection Pipeline

### Multi-Model Ensemble Approach
SVOD uses a **weighted voting system** across 4+ detection methods:
1. **YOLOv10 Person Detection** (PRIMARY, REQUIRED) - `yolov10n.pt` model via ultralytics
2. **DNN Face Detection** - OpenCV DNN with Caffe models (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`)
3. **Facial Landmarks** - LBF landmark detection (`lbfmodel.yaml`) for precise orientation
4. **MobileNet Classification** - OpenVINO models (`mobilenet-v2.xml/.bin`) - optional enhancement
5. **Haar Cascade Face Detection** - OpenCV fallback detector

### Key Detection Logic (OrientationDetector class)
- `process_video_unified()` - Main processing loop analyzing frames
- `determine_frame_orientation()` - Per-frame analysis combining all models
- `determine_final_orientation()` - Weighted voting across frame results
- Frame analysis uses **aspect ratio awareness**: landscape videos with wide detections suggest portrait content rotated sideways

### Landscape-Portrait Content Detection
**Critical pattern**: Landscape videos may contain portrait content (e.g., `P2170127.mp4`). Detection relies on:
- Detection box aspect ratios (wide boxes in landscape = portrait content)
- Face positioning (left/right sides suggest rotation)
- Strong bias values (15.0+) for decisive verdicts
- Environment controls: `SVOD_YOLO10_DECISION_FACTOR`, `SVOD_YOLO10_REDUCE_UNCERTAIN`, `SVOD_FORCE_DECISION`

## Development Workflows

### Testing Philosophy: Two Distinct Ecosystems
**`tests/` directory** - Automated pytest suite (18 test files):
- Unit tests: `test_basic.py`, `test_core_detection.py`, `test_face_detection.py`
- Integration tests: `test_integration.py`, `test_model_integration.py`
- Run with: `pytest tests/` or `make test`
- Coverage requirement: 15% minimum (see `pyproject.toml`)

**`testing/` directory** - Manual real-video testing scripts (MANDATORY for video validation):
- `standard_single_test.py` - Individual video testing
- `standard_batch_test.py` - Batch folder validation with reference data comparison
- `standard_performance_test.py` - Performance benchmarking
- **NEVER create new test scripts** - always use these 3 standard scripts

### Reference Validation System
**`reference_orientations.csv`** - Ground truth dataset (16 videos with known orientations):
- Loaded via `detector.load_reference_data()` or `--reference` CLI flag
- Format: `filename,expected_orientation,confidence,notes`
- Used by `validate_against_reference()` to verify detection accuracy
- **Achievement**: v4.23.0 reached 100% accuracy on reference dataset (8/8 files tested)
- Batch testing automatically compares results and shows accuracy metrics
- Direction validation: Verifies rotation suggestions (clockwise/counterclockwise) match expected

### Build & Development Commands
Use **Makefile** for all common tasks:
```bash
make install    # Install dependencies + pre-commit hooks
make format     # Black formatting (line length 100)
make lint       # Flake8 (ignore E203, W503)
make test       # Run pytest suite
make check      # format + lint + test
make clean      # Remove __pycache__, .coverage, etc.
make build      # Build distribution with python -m build
```

**PowerShell commands** (Windows primary platform):
```powershell
# Environment controls for optimization tuning
$env:SVOD_YOLO10_DECISION_FACTOR='1.02'      # Aggressiveness (1.02-1.05)
$env:SVOD_YOLO10_REDUCE_UNCERTAIN='1'        # Smart fallbacks (0/1)
$env:SVOD_FORCE_DECISION='1'                 # Force decisions (0/1)
$env:SVOD_YOLO10_CONF='0.4'                  # Person detection threshold
$env:SVOD_YOLO10_FACE_CONF='0.55'            # Face confidence override
```

### Python Version Constraints
- **Required:** Python 3.11-3.12
- **Why:** `omz_downloader` (OpenVINO Model Zoo tools) fails on 3.13+ due to NumPy compilation
- **Code location:** `check_system_requirements()` enforces version check

## Project-Specific Conventions

### Security Hardening Patterns
Security is deeply integrated throughout the codebase:
- **Path validation:** `check_system_requirements()` validates paths, rejects directory traversal (`..`), null bytes, excessive lengths (>4096 chars)
- **Resource limits:** `--max-files 1000`, `--max-depth 10` for batch processing
- **Time limits:** Default 30s analysis to prevent resource exhaustion
- **Input sanitization:** All user paths go through `os.path.abspath()` + suspicious pattern checks

### Model Auto-Download System
`download_model_files()` and `install_required_packages()` handle all dependencies:
- Models download on first run from GitHub/OpenVINO repos
- MobileNet requires `omz_downloader` + `omz_converter` (OpenVINO dev tools)
- **Apple Silicon exception:** MobileNet support limited, core algorithms continue without it
- Validation: `validate_model_file()` checks file size and content to reject HTML error pages

### Statistics Tracking Pattern
`OrientationDetector.stats` dictionary tracks all detection metrics:
- `frames_processed`, `faces_detected_dnn`, `persons_detected_yolo10`
- `forced_landscape_portrait_incorrect` - count of landscape videos with portrait content
- Access via `get_statistics()`, display with `print_statistics()`

### Enum-Based Orientation System
```python
class VideoOrientation(Enum):
    CORRECT = "CORRECT - Humans are upright"
    INCORRECT = "INCORRECT - Humans are sideways/rotated"
    UNCERTAIN = "UNCERTAIN - Cannot determine orientation"
```
Used consistently throughout for type safety and clear semantics.

## Critical Dependencies & Model Files

### Required Files (All auto-downloaded)
- `yolov10n.pt` - YOLOv10 nano model (ultralytics auto-downloads)
- `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel` - DNN face detector
- `lbfmodel.yaml` - Facial landmark model
- `coco.names` - YOLO class names
- `mobilenet-v2.xml`, `mobilenet-v2.bin` - MobileNet OpenVINO models (optional)

### Pinned Package Versions (requirements.txt)
Key pins for reproducibility:
- `opencv-contrib-python==4.8.1.78` (contrib required for facial landmarks)
- `numpy==1.26.4` (v2.0+ incompatible with some dependencies)
- `ultralytics==8.3.196` (YOLOv10 support)
- `openvino==2024.6.0` (model optimization)

## Key Files & Directories

- **`video_orientation_detector.py`** - Monolithic 6,778-line core module (all logic)
- **`pyproject.toml`** - Project metadata, entry point: `svod` CLI command
- **`reference_orientations.csv`** - Ground truth dataset (16 videos) for validation testing
- **`performance_baselines/`** - Version performance benchmarks (v4.17.0 through v4.23.0) for regression testing
- **`YOLOV10_UPGRADE.md`** - Detailed YOLOv10 optimization documentation, 100% validation milestone
- **`HOUSEKEEPING_PLAN.md`** - Technical debt and cleanup tracking

## Video Verification Strategy

### Multi-Level Validation Approach
1. **Reference Dataset** - 16 curated videos in `reference_orientations.csv` with known orientations
   - Includes challenging cases: `P2170127.mp4` (landscape with portrait content)
   - Mix of correct/incorrect orientations, various rotation types (clockwise/counterclockwise)
   
2. **Batch Testing** - `standard_batch_test.py` compares against reference data
   - Reports orientation accuracy (CORRECT vs INCORRECT match rate)
   - Validates rotation direction suggestions (clockwise vs counterclockwise)
   - Example metrics: "Reference Accuracy: 8/8 (100%)", "Direction Accuracy: 6/8 (75%)"

3. **Performance Baselines** - Historical accuracy tracking across versions
   - Each version has baseline file in `performance_baselines/`
   - v4.23.0 achieved 100% reference validation (documented milestone)

4. **Real-World Folders** - Manual testing with `Good_Examples/` and `Bad_Examples/`
   - Good_Examples: Expected >95% CORRECT classification
   - Bad_Examples: Expected mix of INCORRECT and UNCERTAIN (challenging cases)

## Common Pitfalls & Solutions

1. **YOLOv10 "not available" error**: Install ultralytics first: `pip install ultralytics`
2. **MobileNet download fails**: Expected on Apple Silicon - core detection still accurate
3. **NumPy v2.0 issues**: Pin to `numpy==1.26.4` (specified in requirements.txt)
4. **"omz_downloader not found"**: Requires Python 3.11-3.12, install `openvino-dev`
5. **UNCERTAIN verdicts**: Tune environment variables (see PowerShell commands above)

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
- **Version docs**: `YOLOV10_UPGRADE.md` for architectural decisions, `HOUSEKEEPING_PLAN.md` for technical debt
