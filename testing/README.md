# Manual Testing Scripts

This directory contains **manual testing scripts** for real video testing, performance benchmarking, and regression testing.

## 📋 Standard Test Scripts

### 1. `standard_single_test.py` - Single Video Testing
```bash
python testing/standard_single_test.py "path/to/video.mp4" [--time-limit 10]
```
- **Purpose**: Test individual video files with detailed analysis
- **Use Cases**: Quick validation, debugging specific videos, verification
- **Output**: Result, confidence, recommendation, and a match against `reference_orientations.csv` (exit code 2 on mismatch)

### 2. `standard_batch_test.py` - Batch Video Testing  
```bash
python testing/standard_batch_test.py --time-limit 10                         # all 37 reference videos
python testing/standard_batch_test.py --folder Good_Examples --time-limit 10   # 22 correct videos
python testing/standard_batch_test.py --folder Bad_Examples --time-limit 10    # 15 incorrect videos
```
- **Purpose**: Validate `C:\Users\boris\Good_Examples` and `C:\Users\boris\Bad_Examples` against `reference_orientations.csv`
- **Use Cases**: Regression testing after any detection change, accuracy assessment
- **Output**: Per-video match, orientation accuracy, direction accuracy, and mismatch list
- **Runtime**: Roughly 15–30 minutes for all 37 videos on CPU

### 3. `standard_performance_test.py` - Performance Benchmarking
```bash
python testing/standard_performance_test.py [--test-video PATH] [--iterations 3] [--save-baseline]
```
- **Purpose**: Performance testing and benchmarking
- **Use Cases**: Version comparisons, optimization validation, system benchmarking
- **Output**: Processing time, FPS, detection counts, memory usage; `--save-baseline` writes to `performance_baselines/`

## ⚠️ **MANDATORY USAGE**

**These are the ONLY testing scripts that should be used for manual video testing.**

- ❌ **DO NOT** create new test scripts
- ❌ **DO NOT** use ad-hoc testing approaches  
- ✅ **ALWAYS** use one of these 3 standard scripts
- ✅ **ENSURE** consistent testing methodology

## 🔄 **Distinction from `tests/` Directory**

- **`tests/` directory**: Automated unit/integration tests (pytest)
- **`testing/` directory**: Manual testing scripts for real videos

Both are essential but serve different purposes in the testing ecosystem.