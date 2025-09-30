# Manual Testing Scripts

This directory contains **manual testing scripts** for real video testing, performance benchmarking, and regression testing.

## 📋 Standard Test Scripts

### 1. `standard_single_test.py` - Single Video Testing
```bash
python testing/standard_single_test.py "path/to/video.mp4"
```
- **Purpose**: Test individual video files with detailed analysis
- **Use Cases**: Quick validation, debugging specific videos, verification
- **Output**: Detailed results with confidence, rotation angle, processing time

### 2. `standard_batch_test.py` - Batch Video Testing  
```bash
python testing/standard_batch_test.py "path/to/video/folder"
```
- **Purpose**: Test multiple videos in a directory
- **Use Cases**: Regression testing, bulk validation, accuracy assessment
- **Output**: Summary statistics, accuracy rates, problematic videos

### 3. `standard_performance_test.py` - Performance Benchmarking
```bash
python testing/standard_performance_test.py
```
- **Purpose**: Performance testing and benchmarking
- **Use Cases**: Version comparisons, optimization validation, system benchmarking
- **Output**: FPS measurements, detection times, memory usage

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