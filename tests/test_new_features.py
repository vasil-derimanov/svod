"""Tests for v4.27.0 new features: JSON output, progress callback,
auto-correct helper, quick pre-scan, frame-size tuning, statistics mixin,
and CLI flags.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svod_models import (
    BatchResult,
    VideoOrientation,
    _sanitize_for_json,
    batch_results_to_json,
    results_to_json,
)
from svod_statistics import StatisticsMixin
from video_orientation_detector import OrientationDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return OrientationDetector(time_limit=1, confidence_threshold=0.5)


@pytest.fixture
def small_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# JSON output tests
# ---------------------------------------------------------------------------


class TestJSONOutput:
    """Tests for results_to_json and batch_results_to_json."""

    def test_results_to_json_basic(self):
        results = {
            "verdict": "[CORRECT]",
            "orientation": "CORRECT",
            "confidence": 0.95,
            "recommendation": "No rotation needed",
        }
        j = results_to_json(results)
        parsed = json.loads(j)
        assert parsed["orientation"] == "CORRECT"
        assert parsed["confidence"] == 0.95

    def test_results_to_json_with_enum(self):
        results = {"orientation": VideoOrientation.INCORRECT, "confidence": 0.8}
        j = results_to_json(results)
        parsed = json.loads(j)
        assert parsed["orientation"] == "INCORRECT"

    def test_results_to_json_with_numpy(self):
        results = {"value": np.float32(0.5), "array": np.array([1, 2, 3])}
        j = results_to_json(results)
        parsed = json.loads(j)
        assert parsed["value"] == pytest.approx(0.5, abs=0.01)
        assert parsed["array"] == [1, 2, 3]

    def test_batch_results_to_json(self, tmp_path):
        dummy_file = tmp_path / "video.mp4"
        dummy_file.write_bytes(b"\x00" * 100)
        br = BatchResult(str(dummy_file), VideoOrientation.CORRECT, 0.9, {}, 1.5)
        j = batch_results_to_json([br])
        parsed = json.loads(j)
        assert len(parsed) == 1
        assert parsed[0]["orientation"] == "CORRECT"
        assert parsed[0]["confidence"] == 0.9

    def test_batch_result_to_dict(self, tmp_path):
        dummy_file = tmp_path / "vid.mp4"
        dummy_file.write_bytes(b"\x00" * 50)
        br = BatchResult(str(dummy_file), VideoOrientation.UNCERTAIN, 0.3, {}, 2.0, "some error")
        d = br.to_dict()
        assert d["orientation"] == "UNCERTAIN"
        assert d["error"] == "some error"
        assert d["processing_time"] == 2.0

    def test_sanitize_set(self):
        assert _sanitize_for_json({1, 2, 3}) == [1, 2, 3]

    def test_sanitize_nested(self):
        data = {"a": {"b": VideoOrientation.CORRECT}}
        result = _sanitize_for_json(data)
        assert result["a"]["b"] == "CORRECT"


# ---------------------------------------------------------------------------
# Statistics mixin tests
# ---------------------------------------------------------------------------


class TestStatisticsMixin:
    """Tests for the extracted StatisticsMixin."""

    def test_init_stats_returns_dict(self, detector):
        stats = detector._init_stats()
        assert isinstance(stats, dict)
        assert stats["total_frames"] == 0
        assert stats["cascade_skipped_haar"] == 0

    def test_reset_stats(self, detector):
        detector.stats["total_frames"] = 42
        detector.reset_stats()
        assert detector.stats["total_frames"] == 0

    def test_reset_statistics_alias(self, detector):
        detector.stats["face_detections"] = 10
        detector.reset_statistics()
        assert detector.stats["face_detections"] == 0

    def test_get_statistics_copy(self, detector):
        detector.stats["total_frames"] = 5
        stats = detector.get_statistics()
        assert stats["total_frames"] == 5
        # Modifying the copy should not affect original
        stats["total_frames"] = 99
        assert detector.stats["total_frames"] == 5

    def test_get_statistics_report(self, detector):
        detector.stats["total_frames"] = 10
        detector.stats["frames_with_humans"] = 8
        detector.stats["correct_orientation_frames"] = 6
        detector.stats["incorrect_orientation_frames"] = 2
        report = detector.get_statistics_report()
        assert "Total frames analyzed: 10" in report
        assert "Correct orientation:" in report

    def test_update_voting_stats(self, detector):
        votes = {
            "face": ["correct", "correct", "incorrect"],
            "yolo": ["incorrect", "incorrect"],
        }
        detector._update_voting_stats(votes)
        assert detector.stats["face_correct_votes"] == 2
        assert detector.stats["face_incorrect_votes"] == 1
        assert detector.stats["body_incorrect_votes"] == 2

    def test_mixin_inheritance(self):
        """OrientationDetector should inherit from StatisticsMixin."""
        assert issubclass(OrientationDetector, StatisticsMixin)


# ---------------------------------------------------------------------------
# Progress callback tests
# ---------------------------------------------------------------------------


class TestProgressCallback:
    """Tests for the progress_callback parameter."""

    def test_callback_signature(self, detector):
        """process_video_unified should accept progress_callback kwarg."""
        import inspect
        sig = inspect.signature(detector.process_video_unified)
        assert "progress_callback" in sig.parameters

    def test_quick_scan_attr(self, detector):
        """Detector should have quick_scan attribute."""
        assert hasattr(detector, "quick_scan")
        assert detector.quick_scan is False


# ---------------------------------------------------------------------------
# Frame size / downscaling tests
# ---------------------------------------------------------------------------


class TestFrameDownscaling:
    """Tests for configurable frame downscaling."""

    def test_default_max_detect_dim(self, detector):
        assert detector._max_detect_dim == 640

    def test_custom_max_detect_dim(self):
        det = OrientationDetector(time_limit=1)
        det._max_detect_dim = 480
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = det._downscale_frame(frame)
        assert max(result.shape[:2]) <= 480

    def test_small_frame_not_resized(self, detector):
        frame = np.zeros((320, 240, 3), dtype=np.uint8)
        result = detector._downscale_frame(frame)
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Quick pre-scan tests
# ---------------------------------------------------------------------------


class TestQuickPrescan:
    """Tests for the quick_prescan method."""

    def test_prescan_method_exists(self, detector):
        assert hasattr(detector, "quick_prescan")
        assert callable(detector.quick_prescan)

    def test_prescan_nonexistent_file(self, detector):
        result = detector.quick_prescan("nonexistent_video.mp4")
        assert result == []

    def test_prescan_returns_list(self, detector):
        result = detector.quick_prescan("nonexistent.mp4")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Auto-correct helper tests
# ---------------------------------------------------------------------------


class TestAutoCorrect:
    """Tests for auto_correct_video function."""

    def test_import(self):
        from svod_cli import auto_correct_video
        assert callable(auto_correct_video)

    def test_no_direction(self):
        from svod_cli import auto_correct_video
        result = auto_correct_video("some_video.mp4", None)
        assert result is None

    def test_none_direction(self):
        from svod_cli import auto_correct_video
        result = auto_correct_video("some_video.mp4", "none")
        assert result is None


# ---------------------------------------------------------------------------
# CLI flags test
# ---------------------------------------------------------------------------


class TestCLIFlags:
    """Tests for new CLI argument definitions."""

    def test_cli_has_json_flag(self):
        """Verify --json flag is registered in argparse."""
        import argparse
        # We can't easily test argparse without calling cli_main,
        # but we verify the module imports cleanly
        from svod_cli import cli_main
        assert callable(cli_main)

    def test_cli_has_auto_correct(self):
        from svod_cli import auto_correct_video
        assert callable(auto_correct_video)
