# SVOD - Smart Video Orientation Detector

## Project Overview
SVOD detects video orientation (CORRECT/INCORRECT/UNCERTAIN) using AI-powered computer vision. `video_orientation_detector.py` is the public entry point. The detector class is `svod_detector.OrientationDetector`. Setup and model downloads live in `svod_bootstrap.py`. Shared flags live in `svod_state.py`.

**Current Version:** 4.28.0 - Project Structure Release (24 November 2026)
**Previous Milestones:**
- v4.27.1 - Reference accuracy fix (`P7061239.mp4` body-vote override)
- v4.27.0 - Developer Experience Release (ONNX Runtime, parallel inference, CLI refactor)
- v4.26.0 - Performance Optimization Release (adaptive cascade, early exit)
- v4.25.0 - YuNet and 100% direction accuracy
- v4.23.0 - 100% orientation accuracy on the reference dataset

### v4.28.0 Structure
- `video_orientation_detector.py` is the public import path and the `--version` fast path (93 lines). It re-exports the CLI helpers and the detector.
- `svod_detector.py` holds `OrientationDetector`. Do not rewrite method bodies while moving them. A bad wrap changed `box.conf` into `box.con` and broke a test.
- `svod_bootstrap.py` installs dependencies, prints console output, and downloads models. Dev tools (black, flake8, pytest, pre-commit) are not auto-installed. `onnxruntime` is required. MediaPipe and `openvino-dev` are optional.
- `svod_state.py` holds `YOLO_AVAILABLE`, `TQDM_AVAILABLE`, and `mobilenet_required_override`. Assigning `vod.YOLO_AVAILABLE` on the facade updates this module.
- Only `main` remains on the remote. Do not revive `feature/boril` or other pre-YOLOv11 scripts.
- Lessons from this release: `LESSONS.md`

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

### v4.27.1 Fixes
- **Landscape body-vote override** (`calculate_final_verdict()`): landscape video (aspect > 1.2) with ≥5 body votes, `body_incorrect > body_correct * 2`, faces ≥ body votes, and frame correct-ratio < 0.90 → INCORRECT. Fixed `P7061239.mp4` (27/32 frames looked correct but body votes were 10 incorrect vs 4 correct). Regression tests: `TestBodyVoteOverride` in `tests/test_core_detection.py`
- **No-display annotation skip**: full mode skips `annotate_frame()` when neither display nor output file is requested (removes unneeded per-frame work; speed gain not confirmed by the 3-run baseline)
- **Testing script fixes**: performance test reads `statistics`/`detection_types` (FPS and detection counts were always 0); single test resolves the CSV from the repo root and reads `statistics`; all scripts print real newlines instead of literal `\n`

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
**`tests/` directory** - Automated pytest suite (20 test files, 216 tests, including `tests/test_cli_security.py`):
- Unit tests: `test_basic.py`, `test_core_detection.py`, `test_face_detection.py`
- Integration tests: `test_integration.py`, `test_model_integration.py`
- Feature tests: `test_new_features.py`, `test_performance_optimizations.py`
- Run with: `pytest tests/` or `make test`
- Coverage requirement: 30% minimum across `video_orientation_detector`, `svod_detector`, `svod_bootstrap`, `svod_cli`, `svod_models`, `svod_statistics`, and `svod_state`. Current: 32.86% (2026-11-24)

**`testing/` directory** - Manual real-video testing scripts (videos live in `C:\Users\boris\Good_Examples` and `C:\Users\boris\Bad_Examples`):
- `standard_single_test.py <video> --time-limit 10` - Individual video, compared against the CSV (exit 2 on mismatch)
- `standard_batch_test.py --time-limit 10 [--folder Good_Examples|Bad_Examples]` - Full reference validation (~30 min for all 37)
- `standard_performance_test.py [--test-video PATH] [--save-baseline]` - Performance benchmarking
- **NEVER create new test scripts** - always use these 3 standard scripts

### Reference Validation System
**`reference_orientations.csv`** - Gold-standard ground truth (37 videos: 22 correct, 15 incorrect):
- Format: `filename,expected_orientation,confidence,notes` (direction is in `notes`)
- **NEVER modify reference_orientations.csv without explicit user approval** - fix detection code instead
- Current accuracy: **100% orientation (37/37), 100% direction (37/37)** (rechecked 2026-11-24 with `standard_batch_test.py --time-limit 10`)
- **After ANY detection change, rerun all 37 videos** with `standard_batch_test.py` before claiming success
- **Never push to git before the full reference validation and `pytest tests/` pass**

### Build & Development Commands
```bash
make install    # Install dependencies + pre-commit hooks
make format     # Black formatting (line length 100)
make lint       # Flake8 (config in .flake8 file)
make test       # Run pytest suite
make test-ci    # Pytest with the 30% coverage gate
make check      # format + lint + test
make clean      # Remove __pycache__, .coverage, etc.
make build      # Build distribution
```

CI (`.github/workflows/cross-platform-test.yml`) installs `requirements.txt` and runs pytest on Ubuntu 3.11, Ubuntu 3.12, Windows 3.11, and macOS Apple Silicon 3.11. Intel macOS (`macos-13`) runs `python video_orientation_detector.py --version` only. PyTorch 2.8 publishes no macOS x86_64 wheel. The reference videos are not in the repo, so CI does not run the 37-video set.

**Required PowerShell environment for testing:**
```powershell
$env:PYTHONIOENCODING='utf-8'
$env:TF_CPP_MIN_LOG_LEVEL='3'
```

### Python Version Constraints
- **Required:** Python 3.11-3.12 (3.13+ not supported due to NumPy/omz_downloader issues)

## Project-Specific Conventions

### Security Hardening
- **Path validation:** Rejects a path component that is `..` or a longer run of dots. A file name such as `clip..final.mp4` is allowed. Also rejects null bytes and paths longer than 4096 characters.
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

- **`video_orientation_detector.py`** - Public entry point and `--version` fast path
- **`svod_detector.py`** - `OrientationDetector` class
- **`svod_bootstrap.py`** - Dependency install, console output, and model downloads
- **`svod_state.py`** - Mutable flags (`YOLO_AVAILABLE`, `TQDM_AVAILABLE`, `mobilenet_required_override`)
- **`svod_cli.py`** - CLI argument parsing, validation, and entry points
- **`svod_statistics.py`** - `StatisticsMixin` for statistics collection and reporting
- **`svod_models.py`** - Data models and JSON serialization (`results_to_json`, `batch_results_to_json`)
- **`pyproject.toml`** - Project metadata, `svod` CLI entry point
- **`.flake8`** - Authoritative lint config. Tests and `testing/` are included. `E501` stays ignored because Black does not wrap long debug strings. There is no per-file ignore list.
- **`Dockerfile`** - Python 3.12 image, pinned requirements, non-root user
- **`LESSONS.md`** - Memo for the v4.28.0 structure work
- **`reference_orientations.csv`** - Gold-standard ground truth (37 videos)
- **`inspect_rotation.py`** - Developer utility for debugging rotation direction
- **`cleanup.ps1` / `cleanup.py`** - Safe cleanup of caches, coverage output, and legacy models
- **`performance_baselines/`** - Version performance benchmarks
- **`YOLOV10_UPGRADE.md`** - Historical migration documentation (not actively maintained)

## Common Pitfalls & Solutions

1. **OpenCV conflict**: Only install `opencv-contrib-python` — never `opencv-python` alongside it
2. **Stale __pycache__**: Always `Remove-Item -Recurse -Force __pycache__` before testing code changes
3. **Batch test encoding**: Always set `$env:PYTHONIOENCODING='utf-8'` before running
4. **NumPy v2.0**: Pin to `numpy==1.26.4`
5. **MediaPipe unavailable**: SVOD injects a lightweight stub; install `mediapipe` for full support
6. **YOLOv11 "not available" error**: Install ultralytics: `pip install "ultralytics>=8.3.0"`
7. **MobileNet download fails**: Expected on Apple Silicon — core detection still accurate
8. **"omz_downloader not found"**: Requires Python 3.11-3.12, install `openvino-dev`
9. **PowerShell exit code 1 with `*>` / `2>&1`**: native stderr (MediaPipe/TF warnings) triggers `NativeCommandError`; check the pytest summary or `$LASTEXITCODE`, not the terminal exit status
10. **`make clean` on Windows**: uses Unix `find`; run `.\cleanup.ps1` instead
11. **Do not auto-wrap `svod_detector.py`**: a line wrap shortened `conf` to `con` and split f-strings into calls. Compare method ASTs with the previous file before treating a move as behavior-preserving.

## Real-World Testing

- **Good_Examples/**: 22/22 CORRECT; **Bad_Examples/**: 15/15 INCORRECT with correct direction (rechecked 2026-11-24, v4.28.0, `--time-limit 10`)
- Hard cases to watch after detection changes: `P7061239.mp4` (landscape body-vote override), `P2170127.mp4` (landscape-portrait content), `VID_20200907_202511.mp4` (rotation metadata)
- Performance baseline is still v4.27.1 (`performance_baselines/performance_v4_27_1_20260928_014100.txt`: 111.4s avg, 0.7 FPS, 1072.6MB peak on `20150911_221520.mp4`). v4.28.0 did not change detection math, so that baseline was not replaced.

## Adding New Features

When extending detection capabilities:
1. Add new detection method to `OrientationDetector` class
2. Integrate into `determine_frame_orientation()` voting logic
3. Add statistics tracking to `self.stats` dictionary
4. Update `print_results()` and `svod_statistics.StatisticsMixin` to display new metrics
5. Add unit tests to `tests/test_*.py` AND manual tests with `testing/standard_*_test.py`
6. Update `README.md` usage examples

## Documentation Standards

- **README.md**: User-facing documentation with CLI examples and version history
- **Code comments**: Inline explanations for complex detection logic
- **Docstrings**: All public methods have detailed docstrings with Args/Returns
- **Version bumps**: update `__version__`/`__release_*` in `video_orientation_detector.py` and `__init__.py`, `version` in `pyproject.toml`, the `requirements.txt` header, README version history, and this file. The other modules do not carry their own version.
