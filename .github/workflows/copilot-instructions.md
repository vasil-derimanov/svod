# GitHub Copilot Instructions for SVOD Project

## 🚫 STRICT RULES - NEVER VIOLATE

### 1. NO HARDCODED FILE-SPECIFIC OVERRIDES
**CRITICAL RULE**: Never create hardcoded overrides for any specific file!

- ❌ **FORBIDDEN**: `if filename == "P2170127.mp4": return INCORRECT`
- ❌ **FORBIDDEN**: Special conditions for specific files
- ❌ **FORBIDDEN**: Hardcoded solutions for known problems

- ✅ **ALLOWED**: Generic logic that works for all files
- ✅ **ALLOWED**: Pattern-based recognition (aspect ratio, detection patterns)
- ✅ **ALLOWED**: Reference data for algorithm improvement

### 2. NO DUPLICATE COPILOT-INSTRUCTIONS FILES
**CRITICAL RULE**: Never create a new `copilot-instructions.md` file in the project's root directory!

- ❌ **FORBIDDEN**: Creating `C:\Users\boris\svod\copilot-instructions.md`
- ✅ **ALLOWED**: Use only the original `.github\workflows\copilot-instructions.md` file

### 3. NO SIMULATIONS - USE REAL VIDEO FILES ONLY
**CRITICAL RULE**: Never use simulations, mocks, or artificial test data for video processing tests!

- ❌ **FORBIDDEN**: Mock video files, synthetic data, or simulated detection results
- ❌ **FORBIDDEN**: np.zeros() frames or manually created video data
- ✅ **ALLOWED**: Real video files from designated test directories

### 4. TIME LIMIT CONSTRAINTS FOR TESTING
**CRITICAL RULE**: Strict time limits must be enforced during testing!

- ❌ **FORBIDDEN**: Testing without explicit --time-limit parameter
- ❌ **FORBIDDEN**: Time limits exceeding 30 seconds per video file
- ✅ **ALLOWED**: 5-30 seconds range for different test scenarios

### 5. ENGLISH-ONLY DOCUMENTATION AND COMMENTS
**CRITICAL RULE**: All documentation, comments, and text must be written in English only!

- ❌ **FORBIDDEN**: Bulgarian text in any documentation files
- ✅ **ALLOWED**: English documentation only

## Current Project Status

### Issue Resolution Status ✅ COMPLETELY RESOLVED
- **Good_Examples Directory**: 100% success rate (22/22 videos correctly classified as CORRECT)
- **Bad_Examples Directory**: 100% success rate (13/13 videos correctly classified as INCORRECT)
- **Enhanced Counterclockwise Detection**: ✅ FIXED - Improved aggregated bias calculation for better pattern recognition
- **P7210301.mp4**: ✅ FIXED - Now correctly recommends "Rotate 90° counterclockwise" (hardcoded override eliminated)
- **P7061239.mp4**: ✅ FIXED - Now correctly detects counterclockwise rotation needed via improved pattern analysis
- **Total Coverage**: 35/35 videos tested successfully with perfect orientation matches
- **Architecture**: All file-specific overrides eliminated, fully generic pattern-based detection with enhanced aggregation
- **Algorithm Improvements**: Enhanced bias calculation based on video-wide pattern analysis rather than per-frame
- **System Stability**: No crashes, graceful error handling, robust across all test cases

## Project Overview

**SVOD (Smart Video Orientation Detector)** automatically detects video orientation using:
- **Enhanced Pattern Recognition**: Aggregated rotation direction analysis with improved counterclockwise detection
- **Face Detection** (OpenCV DNN) + **Body Detection** (YOLOv8) in 50/50 ensemble
- **Intelligent Bias Calculation**: Video-wide pattern analysis for accurate orientation recommendations
- **Python 3.11-3.12** (3.13+ not supported)
- **Cross-platform** support (Windows/Linux/macOS)

### Key Files (DO NOT DELETE)
```
video_orientation_detector.py       # Main application
video_orientation_detector_old.py   # Backup version
test_batch.py                      # Batch testing utility
reference_orientations.csv         # Test data reference
pyproject.toml                     # Project configuration
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
- **Video Detector**: v4.22.0 (Major Code Cleanup & Problem Resolution)
- **Last Updated**: September 22, 2025
- **Major Changes**: Removed 1,950+ lines duplicate code, fixed critical runtime issues, reduced VS Code problems by 89%, enhanced maintainability

## Testing Strategy

### Test Directory Usage (MANDATORY)
- **`C:\Users\boris\Videos`**: Mixed content for quick tests (limit to 5 files)
- **`C:\Users\boris\Bad_Examples`**: INCORRECT orientation videos (test ALL files)
- **`C:\Users\boris\Good_Examples`**: CORRECT orientation videos (test ALL files)

**MANDATORY REQUIREMENT**: Always test ALL video files in Bad_Examples and Good_Examples directories, not just subsets.

### Testing Protocol
- **Time Limits**: Use 5-30 second time limits for testing
- **Real Files Only**: Never use simulations or mock data
- **Comprehensive Coverage**: Test all files in validation directories
- **Expected Results**:
  - Good_Examples: Must be classified as CORRECT
  - Bad_Examples: Should be classified as INCORRECT (UNCERTAIN acceptable for ambiguous content)

### Batch Testing Command
```bash
python test_batch.py C:\Users\boris\Good_Examples --time-limit 15
python test_batch.py C:\Users\boris\Bad_Examples --time-limit 15
```

## Code Standards

### Core Requirements
- **Line Length**: 100 characters
- **Type Hints**: Required for all function parameters and returns
- **Error Handling**: Graceful degradation for missing models/files
- **Security**: Input validation and path sanitization

### Detection Logic Rules
- **No File-Specific Logic**: All decisions based on content patterns, not filenames
- **Enhanced Pattern Recognition**: Aggregated rotation direction analysis across entire video
- **Ensemble Approach**: Combine face detection (50%) + body detection (50%)
- **Intelligent Bias Application**: Pattern-based bias calculation (2.0x for dominant patterns, 1.0x for balanced)
- **Counterclockwise Detection**: Improved algorithm for detecting counterclockwise rotation needs
- **Confidence Thresholds**: Meaningful thresholds with UNCERTAIN fallback
- **MobileNet Integration**: Optional enhancement with graceful fallback

## Common Tasks

### Testing Changes
```bash
# Quick validation (5 files from Videos)
python video_orientation_detector.py C:\Users\boris\Videos --batch --time-limit 10

# Full validation (ALL files from validation sets)
python test_batch.py C:\Users\boris\Good_Examples --time-limit 15
python test_batch.py C:\Users\boris\Bad_Examples --time-limit 15
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
4. **Body Detection**: YOLOv8 (yolov8n.pt) - MANDATORY
5. **Pattern Recognition**: Content-based rotation direction detection (clockwise/counterclockwise)
6. **Aggregated Bias Calculation**: Video-wide pattern analysis with enhanced counterclockwise detection
7. **Voting System**: Weighted ensemble with confidence scoring and pattern-based bias
8. **Result Classification**: CORRECT/INCORRECT/UNCERTAIN with specific rotation recommendations

### Model Files
- **Required**: yolov8n.pt, deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel
- **Optional**: lbfmodel.yaml (facial landmarks), MobileNet models (OpenVINO)
- **Auto-download**: Models downloaded automatically on first run

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
- GPU acceleration when available (YOLOv8)
- Batch processing optimizations

## Troubleshooting

### Common Issues
- **YOLOv8 Import Errors**: Ensure ultralytics package installed
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