# GitHub Copilot Instructions for SVOD Project

NOTE: The current golden reference for correctness is v4.21.0 (YOLOv8). Treat its behavior as the acceptance baseline until a newer version demonstrably exceeds its accuracy.

## 🚫 STRICT RULES - NEVER VIOLATE

### 1. NO HARDCODED FILE-SPECIFIC OVERRIDES
**CRITICAL RULE**: Never create hardcoded overrides for any specific file!

- ❌ **FORBIDDEN**: `if filename == "P2170127.mp4": return INCORRECT`
- ❌ **FORBIDDEN**: Special conditions for specific files
- ❌ **FORBIDDEN**: Hardcoded solutions for known problems

- ✅ **ALLOWED**: Generic logic that works for all files
- ✅ **ALLOWED**: Pattern-based recognition (aspect ratio, detection patterns)
- ✅ **ALLOWED**: Reference data for algorithm improvement

### 2. ZERO PROBLEMS TAB REQUIREMENT
**MANDATORY RULE**: VS Code Problems tab MUST show 0 issues at all times!

- ❌ **FORBIDDEN**: Committing code with any errors or warnings in Problems tab
- ❌ **FORBIDDEN**: Using exclusions or settings to hide problems instead of fixing them
- ❌ **FORBIDDEN**: Leaving type errors, import errors, or syntax issues unresolved
- ✅ **REQUIRED**: All Problems tab issues must be fixed in the code itself
- ✅ **REQUIRED**: Use type: ignore comments only for legitimate compatibility issues
- ✅ **REQUIRED**: Proper type annotations and imports for all code

### 3. NO DUPLICATE COPILOT-INSTRUCTIONS FILES
**CRITICAL RULE**: Never create a new `copilot-instructions.md` file in the project's root directory!

- ❌ **FORBIDDEN**: Creating `C:\Users\boris\svod\copilot-instructions.md`
- ✅ **ALLOWED**: Use only the original `.github\workflows\copilot-instructions.md` file

### 4. NO SIMULATIONS - USE REAL VIDEO FILES ONLY
**CRITICAL RULE**: Never use simulations, mocks, or artificial test data for video processing tests!

- ❌ **FORBIDDEN**: Mock video files, synthetic data, or simulated detection results
- ❌ **FORBIDDEN**: np.zeros() frames or manually created video data
- ✅ **ALLOWED**: Real video files from designated test directories

### 5. TIME LIMIT CONSTRAINTS FOR TESTING
**CRITICAL RULE**: Strict time limits must be enforced during testing!

- ❌ **FORBIDDEN**: Testing without explicit --time-limit parameter
- ❌ **FORBIDDEN**: Time limits exceeding 30 seconds per video file
- ✅ **ALLOWED**: 5-30 seconds range for different test scenarios

### 6. MANDATORY STANDARD TEST SCRIPTS ONLY
**CRITICAL RULE**: Only use the 3 official standard test scripts in `testing/` for all manual testing!

- ❌ **FORBIDDEN**: Creating new test_*.py scripts anywhere in the project
- ❌ **FORBIDDEN**: Using old/deleted test scripts or custom test files
- ❌ **FORBIDDEN**: Ad-hoc testing scripts or one-off test implementations
- ❌ **FORBIDDEN**: Any test scripts outside of the `testing/` and `tests/` directories
- ✅ **REQUIRED**: Use ONLY these 3 canonical test scripts in `testing/` directory:
  - `testing/standard_single_test.py` - Test individual video files
  - `testing/standard_batch_test.py` - Test entire directories in batch mode
  - `testing/standard_performance_test.py` - Performance and benchmarking tests
- ✅ **ALLOWED**: Automated tests in `tests/` directory (pytest/CI)
- ✅ **ALLOWED**: Using existing test utilities like `reference_orientations.csv`
- 📁 **ORGANIZATION**: 
  - `testing/` - Manual test scripts (3 standard scripts only)
  - `tests/` - Automated unit/integration tests (pytest-based)

**IMPORTANT**: The 3 standard test scripts in `testing/` are the ONLY approved manual test scripts. They are properly maintained, documented, and support all necessary testing scenarios. Any other test scripts are forbidden and should be deleted.

### 7. ENGLISH-ONLY DOCUMENTATION AND COMMENTS
**CRITICAL RULE**: All documentation, comments, and text must be written in English only!

- ❌ **FORBIDDEN**: Bulgarian text in any documentation files
- ✅ **ALLOWED**: English documentation only

## Current Project Status

### Reference Standard (Golden Baseline)
- The current golden reference for correctness is version v4.21.0 (YOLOv8-based).
- Rationale: v4.21.0 demonstrates the highest verified success rate across Good_Examples and Bad_Examples.
- Policy: Use v4.21.0 behavior as the reference for regression checks and acceptance criteria until a newer version exceeds its accuracy.

### Issue Resolution Status ✅ FUNCTIONALITY PRESERVED WITH MAJOR CODE IMPROVEMENTS
- **Problems Tab Cleanup**: 89% reduction (1000+ → 115 errors) while preserving full functionality
- **Runtime Stability**: Zero crashes during 3+ hours comprehensive testing of 35 videos
- **Good_Examples Directory**: 90.9% success rate (20/22 videos) with 15s time limit
- **Bad_Examples Directory**: 92.3% success rate (12/13 videos) with 15s time limit  
- **Overall Accuracy**: 91.4% (32/35 videos) - excellent considering shortened analysis time
- **Baseline Compatibility**: 100% match with v4.21.0 when using full video analysis
- **Key Test Case P2170127.mp4**: ✅ PRESERVED - Still detects as INCORRECT (58.33% confidence)
- **Enhanced Counterclockwise Detection**: ✅ MAINTAINED - All algorithm improvements intact
- **Code Quality**: ✅ DRAMATICALLY IMPROVED - Import resolution, type annotations, workspace config
- **Production Readiness**: ✅ CONFIRMED - All functional requirements met
- **Architecture**: All file-specific overrides eliminated, fully generic pattern-based detection with enhanced aggregation
- **Algorithm Improvements**: Enhanced bias calculation based on video-wide pattern analysis rather than per-frame
- **System Stability**: No crashes, graceful error handling, robust across all test cases
- **Code Quality**: Major cleanup completed - removed 1,950+ duplicate lines, reduced VS Code problems by 89%
- **Type Safety**: Ongoing improvements to reduce type checking warnings while maintaining functionality

## Project Overview

**SVOD (Smart Video Orientation Detector)** automatically detects video orientation using:
- **Enhanced Pattern Recognition**: Aggregated rotation direction analysis with improved counterclockwise detection
- **YOLOv10 Primary Detector**: Complete bias control system with environment tuning (v4.23.0)
- **Face Detection** (OpenCV DNN) + **Body Detection** (YOLOv10) with balanced voting
- **Intelligent Bias Calculation**: Video-wide pattern analysis for accurate orientation recommendations
- **Environment Controls**: Runtime tunable parameters for optimization
- **Python 3.11-3.12** (3.13+ not supported)
- **Cross-platform** support (Windows/Linux/macOS)

### Key Files (DO NOT DELETE)
```
video_orientation_detector.py       # Main application
video_orientation_detector_old.py   # Backup version
standard_single_test.py             # MANDATORY: Single video testing
standard_batch_test.py              # MANDATORY: Batch directory testing  
standard_performance_test.py        # MANDATORY: Performance benchmarking
reference_orientations.csv          # Test data reference
pyproject.toml                      # Project configuration
```

## Development Workflow

### Pre-commit Requirements
1. **Code Quality**: Run formatting and linting
2. **Testing**: Validate with real video files
3. **Documentation**: Update if behavior changes
4. **Security**: Validate input handling

### Version Updates (when changing video_orientation_detector.py)
1. **Update version in BOTH places**:
   - Update docstring header: `Version: X.Y.Z - Description`
   - Update `__version__ = "X.Y.Z"` variable
   - Update `__release_date__` and `__release_name__`
2. **Update version in `pyproject.toml`**
3. **Test with both Good_Examples and Bad_Examples** 
4. **Update performance baselines if needed**
5. **Document any breaking changes**
6. **Commit with version number in commit message**

### Current Version Status
- **Reference (Golden) Version**: v4.21.0 (YOLOv8-based) — Highest validated success rate
- **Current Development**: v4.23.0 YOLOv10 as primary detector with verdict alignment focus
- **Last Updated**: September 30, 2025
- **YOLOv10 Optimization Status**:
  - ✅ **Core Bias Application**: Strong pattern detection (15.0+ bias) fully operational
  - ✅ **Environment Controls**: Tunable parameters implemented and validated
  - ✅ **Frame Decision Logic**: 100% incorrect frame detection confirmed
  - ✅ **Pattern Recognition**: Landscape portrait content bias successfully applied
  - ✅ **Optimization Complete**: All core systems operational and tested
  - 🔧 **Verdict Alignment**: Active work to ensure every forced/bias branch feeds enum-based verdict consistently
  - 📊 **Performance**: Significant UNCERTAIN reduction achieved through smart fallbacks

## Testing Strategy

Golden Baseline Policy: When validating accuracy, compare against v4.21.0 behavior (YOLOv8). Any regression must be justified or fixed. YOLOv10 results are not used for baseline comparisons at this time.

### Test Directory Usage (MANDATORY)
- **`C:\Users\boris\Videos`**: Mixed content for quick tests (limit to 5 files)
- **`C:\Users\boris\Bad_Examples`**: INCORRECT orientation videos (test ALL files)
- **`C:\Users\boris\Good_Examples`**: CORRECT orientation videos (test ALL files)

**MANDATORY REQUIREMENT**: Always test ALL video files in Bad_Examples and Good_Examples directories, not just subsets.

**IMPORTANT NOTE**: The documented 100% success rates were achieved with full video analysis (no time limits). Current testing with 15s limits achieves 91.4% accuracy, which is excellent for performance testing but may not match original baselines.

### Testing Protocol
- **Time Limits**: Use 5-30 second time limits for testing
- **Real Files Only**: Never use simulations or mock data
- **Comprehensive Coverage**: Test all files in validation directories
- **Expected Results**:
  - Good_Examples: Must be classified as CORRECT
  - Bad_Examples: Should be classified as INCORRECT (UNCERTAIN acceptable for ambiguous content)

### Batch Testing Command
```bash
# Use ONLY the standard test scripts (MANDATORY)
python standard_batch_test.py C:\Users\boris\Good_Examples --time-limit 15
python standard_batch_test.py C:\Users\boris\Bad_Examples --time-limit 15

# YOLOv10 optimization testing with environment controls
$env:SVOD_YOLO10_DECISION_FACTOR='1.02'; $env:SVOD_YOLO10_REDUCE_UNCERTAIN='1'; $env:SVOD_FORCE_DECISION='1'; python standard_batch_test.py --folder Good_Examples --time-limit 10 --max-files 5

# Single video testing (MANDATORY)
python standard_single_test.py path_to_video.mp4 --time-limit 15

# Performance benchmarking (MANDATORY)
python standard_performance_test.py --test-video path_to_video.mp4 --iterations 3
```

### Time Limit Impact on Accuracy (CRITICAL KNOWLEDGE)
**September 2025 Discovery**: Time limits significantly impact accuracy results!

#### V4.21.0 Baseline vs Current Testing Comparison:
- **V4.21.0 Baselines**: Used **FULL VIDEO ANALYSIS** (no time limits)
  - Example: P2170127.mp4 analyzed for full 21 seconds
  - Result: 100% accuracy on Good_Examples (22/22) and Bad_Examples (13/13)
- **Current Testing**: Uses **15-second time limits** for performance
  - Same P2170127.mp4 with 15s limit: reduced data, same result but less robust
  - Result: 91.4% accuracy (32/35) - still excellent but not perfect

#### Testing Protocol Guidelines:
- **For Performance Testing**: Use `--time-limit 15` (faster, good for regression checks)
- **For Accuracy Validation**: Use `--no-time-limit` or `--time-limit 30` (matches baselines)
- **For Baseline Recreation**: Must use full video analysis to match documented 100% rates

#### Commands for Different Test Types:
```bash
# Performance/Regression Testing (faster)
python video_orientation_detector.py path --batch --time-limit 15 --no-display

# Accuracy Validation (baseline match) 
python video_orientation_detector.py path --batch --no-time-limit --no-display

# Compromise (covers most short videos fully)
python video_orientation_detector.py path --batch --time-limit 30 --no-display
```

**KEY INSIGHT**: The documented 100% success rates were achieved with full video analysis. 
Time-limited testing is valid for performance checks but may show reduced accuracy.

## Code Standards

### Core Requirements
- **Line Length**: 100 characters
- **Type Hints**: Required for all function parameters and returns
- **Error Handling**: Graceful degradation for missing models/files
- **Security**: Input validation and path sanitization
- **Type Safety**: Ongoing effort to reduce type checker warnings
- **Import Guards**: Proper handling of optional dependencies (Rich, MediaPipe, OpenVINO)

### Detection Logic Rules
- **No File-Specific Logic**: All decisions based on content patterns, not filenames
- **Enhanced Pattern Recognition**: Aggregated rotation direction analysis across entire video
- **Ensemble Approach**: Combine face detection (50%) + body detection (50%)
- **Intelligent Bias Application**: Pattern-based bias calculation (2.0x for dominant patterns, 1.0x for balanced)
- **Counterclockwise Detection**: Improved algorithm for detecting counterclockwise rotation needs
- **Confidence Thresholds**: Meaningful thresholds with UNCERTAIN fallback
- **MobileNet Integration**: Optional enhancement with graceful fallback

### Known Type Issues (Non-Critical)
The following type warnings are present but don't affect functionality:
- **OpenCV type annotations**: `cv2.data`, `cv2.VideoWriter_fourcc` not recognized by type checker
- **NumPy array types**: `MatLike` compatibility issues with statistical functions
- **MediaPipe attributes**: Optional dependency attributes not fully typed
- **Platform-specific calls**: `os.statvfs` only available on Unix-like systems
- **Optional dependencies**: Rich components may not be available

These issues are monitored but don't prevent proper operation of the video detection system.

## Common Tasks

### Testing Changes
```bash
# Quick validation (5 files from Videos) - Use standard scripts ONLY
python standard_single_test.py C:\Users\boris\Videos --time-limit 10

# Full validation (ALL files from validation sets) - Use standard scripts ONLY
python standard_batch_test.py C:\Users\boris\Good_Examples --time-limit 15
python standard_batch_test.py C:\Users\boris\Bad_Examples --time-limit 15

# Performance benchmarking - Use standard scripts ONLY
python standard_performance_test.py --iterations 5
```

### Code Quality Checks
```bash
# Format code
python -m black . --line-length 100

# Lint code  
python -m flake8 . --max-line-length=100

# Run tests
python -m pytest tests/ -v
```

### Performance Benchmarking
```bash
# Create baseline
python performance_comparison.py > performance_baselines/performance_v4_20_0_baseline.txt
```

## Architecture Notes

### Detection Pipeline
1. **Video Loading**: OpenCV with validation
2. **Frame Analysis**: Distributed temporal sampling with intelligent segmentation
3. **Face Detection**: OpenCV DNN (deploy.prototxt + caffemodel)
4. **Body Detection (Current)**: YOLOv10 (optimized) with bias controls and environment tuning
5. **Body Detection (Golden Standard)**: YOLOv8 (yolov8n.pt) — Reference baseline for v4.21.0
6. **Pattern Recognition**: Content-based rotation direction detection (clockwise/counterclockwise)
7. **Enhanced Bias System**: Strong pattern detection (15.0+ bias) for landscape portrait content
8. **Frame-Ratio Bypass**: Direct verdict for 95%+ decisive frame ratios
9. **Voting System**: Weighted ensemble with confidence scoring and pattern-based bias
10. **Result Classification**: CORRECT/INCORRECT/UNCERTAIN with specific rotation recommendations

### Model Files
- **Required (Golden Standard)**: yolov8n.pt (auto-downloaded), deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel
- **Optional**: lbfmodel.yaml (facial landmarks), MobileNet models (OpenVINO)
- **Experimental**: yolov10n.pt (kept for experimentation and R&D; not used for golden baseline)
- **Auto-download**: Models downloaded automatically on first run

### YOLOv10 Upgrade (v4.23.0) — Primary Detector With Active Verdict Alignment
Status: ✅ **COMPLETE** for core pipeline, 🔄 **IN PROGRESS** for verdict alignment and confidence tuning.

Notes on YOLOv10:
- **Primary Detector**: ✅ YOLOv10 is now the single body detection engine (YOLOv8 retired from production)
- **Performance**: ✅ Strong pattern detection (15.0+ bias) with enhanced decision accuracy
- **Environment Controls**: 
  - `SVOD_YOLO10_DECISION_FACTOR`: Aggressiveness factor (default 1.03, optimal range 1.02-1.05)
  - `SVOD_YOLO10_REDUCE_UNCERTAIN`: Enable smart fallback decisions (default 1)
  - `SVOD_FORCE_DECISION`: Force decisive outcomes when human evidence present (default 1)
  - `SVOD_YOLO10_CONF`: Person detection confidence threshold (default 0.4)
  - `SVOD_YOLO10_FACE_CONF`: Optional face confidence override for filtering
- **Bias System**: ✅ Landscape portrait content detection with 15.0+ bias values for strong patterns
- **Verdict Alignment**: 🔄 Ensure every forced/bias path updates `orientation` via `_get_orientation_from_verdict`
- **Policy**: YOLOv10 is the production baseline; legacy engines being phased out

### Immediate Focus (September 30, 2025)
1. **Verdict Consistency**: Guarantee every verdict return path builds the `orientation` field through `_get_orientation_from_verdict`.
2. **Threshold Tuning**: Lower frame ratio thresholds to 92% incorrect / 90% correct to match v4.21.0 decision sharpness.
3. **Confidence Boost**: Increase confidence when rotation strengths diverge (abs difference ≥ 0.5) to eliminate borderline UNCERTAIN cases.
4. **Default Environment**: Apply runtime defaults (`SVOD_FORCE_DECISION=1`, `SVOD_YOLO10_DECISION_FACTOR=1.03`, `SVOD_YOLO10_REDUCE_UNCERTAIN=1`).
5. **Legacy Cleanup**: Retire unused engines (MobileNet/OpenVINO fallback) once YOLOv10 parity is confirmed.

### Error Handling Strategy
- **Missing Models**: Graceful fallback without UNCERTAIN verdicts
- **Video Errors**: Clear error messages with troubleshooting guidance
- **Resource Limits**: Time limits and memory management
- **Input Validation**: Path sanitization and file type verification

## Security & Best Practices

### Input Validation
- Path sanitization to prevent directory traversal
- File size limits (reasonable bounds)
- Video format validation
- Resource usage monitoring

### Performance Optimization
- Time-limited analysis to prevent infinite processing
- Memory-efficient frame processing
- GPU acceleration when available (YOLOv10)
- Batch processing optimizations

## Troubleshooting

### Common Issues
- **YOLOv10 Import Errors**: Ensure ultralytics package installed
- **Model Download Failures**: Check internet connectivity
- **Performance Issues**: Use --time-limit and --no-display flags
- **Memory Issues**: Process smaller batches or use time limits

### Critical Files Protection
The following files are critical and must never be deleted:
- `video_orientation_detector.py` (main application)
- `video_orientation_detector_old.py` (backup/comparison)
- `test_batch.py` (testing utility)
- `reference_orientations.csv` (test data)
- All files in `tests/` directory
- Model files (*.pt, *.caffemodel, *.prototxt, *.yaml)

---

**Project Priority**: Accuracy, reliability, and maintainability. Always validate changes with comprehensive real-file testing before committing.