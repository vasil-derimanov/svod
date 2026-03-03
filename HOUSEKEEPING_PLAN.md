# Project Housekeeping Plan

**Date:** March 2026 (Updated from November 29, 2025)
**Status:** Phase 2 Complete - Documentation and project files cleaned up
**Goal:** Maintain clean project structure and up-to-date documentation

## ✅ Completed Actions (Phase 1)

### Root-Level Test Scripts - CLEANED UP
All redundant test scripts have been removed from root directory:
- ✅ No `test_*.py` files in root (all moved to `tests/` or deleted)
- ✅ No `debug_*.py` files in root (all removed)
- ✅ Clean separation: `tests/` for pytest, `testing/` for manual validation

### Current Clean State
Root directory now contains only essential files:
- **Core Module**: `video_orientation_detector.py` (7,879 lines)
- **Configuration**: `pyproject.toml`, `requirements.txt`, `Makefile`
- **Documentation**: `README.md`, `YOLOV10_UPGRADE.md`, `HOUSEKEEPING_PLAN.md`
- **Utilities**: `cleanup.py`, `cleanup.ps1`, `inspect_rotation.py`
- **Models**: Auto-downloaded AI model files
- **Folders**: `tests/`, `testing/`, `performance_baselines/`, `.github/`

## 🔄 Phase 2: Documentation Maintenance (November 2025)

### Version Updates - ✅ COMPLETED
- [x] Updated `video_orientation_detector.py` to v4.24.0
- [x] Updated `pyproject.toml` to v4.24.0
- [x] Updated `README.md` version history
- [x] Updated `.github/copilot-instructions.md` with current version

### Files to Monitor
1. **`video_orientation_detector_old.py`** - Legacy file, removed during Phase 2 cleanup
   - Status: ✅ Deleted (March 2026)

2. **`comparison_results.json`** - Old comparison data
   - Status: ✅ Deleted - contained failed test data, no longer relevant (March 2026)

3. **Legacy model files** (`yolov10n.pt`, `yolov8n.pt`) - Superseded by YOLOv11
   - Status: ✅ Deleted - active models are `yolo11n.pt` and `yolo11n-pose.pt` (March 2026)

## 📊 Current Project Statistics

### Test Coverage
- **pytest suite**: 17 test files in `tests/`
- **Manual validation**: 3 standard scripts in `testing/`
- **Reference dataset**: 18 videos in `reference_orientations.csv`
- **Performance baselines**: 8 versions tracked (v4.17.0 → v4.23.0)

### Code Quality
- **Main module**: 7,879 lines (monolithic by design)
- **Test coverage**: 15% minimum requirement
- **Code style**: Black formatting (line length 100)
- **Linting**: Flake8 (max line length 120, ignore E203)

## 🎯 Future Housekeeping Tasks

### Priority 1: Archive Management
- [x] ~~Create `archive/` folder for historical files~~ Deleted obsolete files directly
- [x] Removed `video_orientation_detector_old.py` (March 2026)
- [x] Removed old comparison results (March 2026)
- [x] Removed legacy model files `yolov10n.pt`, `yolov8n.pt` (March 2026)

### Priority 2: Documentation Review
- [x] Updated all YOLOv10 references to YOLOv11 across README, copilot-instructions
- [x] Updated line counts and file counts in all documentation
- [x] Fixed `__init__.py` version (was 4.21.0, now 4.24.0)
- [x] Cleaned `.flake8` stale exclusion entries
- [ ] Review `YOLOV10_UPGRADE.md` for outdated migration steps
- [ ] Update any hardcoded paths to use environment variables

### Priority 3: Test Enhancement
- [ ] Add performance baseline for v4.24.0
- [ ] Update `testing/README.md` with latest best practices
- [ ] Document reference validation improvements

## ✅ Best Practices Established

1. **Testing**: Use only `tests/` (pytest) and `testing/` (manual) directories
2. **Version Control**: Update all version references simultaneously
3. **Documentation**: Keep README.md, HOUSEKEEPING_PLAN.md, and copilot-instructions.md in sync
4. **Cleanup**: Run `make clean` regularly to remove temporary files
5. **Validation**: Use `reference_orientations.csv` for accuracy verification

## 📍 Video Sample Locations (Confirmed)
- Good Examples: `C:\Users\boris\Good_Examples` (expected: >95% CORRECT)
- Bad Examples: `C:\Users\boris\Bad_Examples` (expected: mix of INCORRECT/UNCERTAIN)
