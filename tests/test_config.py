import pytest
import numpy as np
from unittest.mock import patch
from video_orientation_detector import OrientationDetector

class TestConfiguration:
    """Test different configuration options"""

    def test_default_configuration(self):
        """Test default configuration values"""
        detector = OrientationDetector()

        assert detector.confidence_threshold == 0.5
        assert detector.time_limit is None
        assert hasattr(detector, 'stats')
        assert isinstance(detector.stats, dict)

    def test_custom_confidence_threshold(self):
        """Test custom confidence threshold"""
        detector_low = OrientationDetector(confidence_threshold=0.3)
        detector_high = OrientationDetector(confidence_threshold=0.8)

        assert detector_low.confidence_threshold == 0.3
        assert detector_high.confidence_threshold == 0.8

        # Test that they work with frames
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(detector_low, 'detect_faces_dnn', return_value=[]):
            with patch.object(detector_low, 'detect_persons', return_value=[]):
                result_low = detector_low.determine_frame_orientation(frame)
                assert result_low is not None

        with patch.object(detector_high, 'detect_faces_dnn', return_value=[]):
            with patch.object(detector_high, 'detect_persons', return_value=[]):
                result_high = detector_high.determine_frame_orientation(frame)
                assert result_high is not None

    def test_custom_time_limit(self):
        """Test custom time limit"""
        detector_quick = OrientationDetector(time_limit=5)
        detector_slow = OrientationDetector(time_limit=60)
        detector_unlimited = OrientationDetector(time_limit=None)

        assert detector_quick.time_limit == 5
        assert detector_slow.time_limit == 60
        assert detector_unlimited.time_limit is None

    def test_combined_configuration(self):
        """Test combined configuration settings"""
        detector = OrientationDetector(
            confidence_threshold=0.7,
            time_limit=30
        )

        assert detector.confidence_threshold == 0.7
        assert detector.time_limit == 30

        # Test that it works
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(detector, 'detect_faces_dnn', return_value=[]):
            with patch.object(detector, 'detect_persons', return_value=[]):
                result = detector.determine_frame_orientation(frame)
                assert result is not None

    def test_extreme_confidence_values(self):
        """Test extreme confidence threshold values"""
        detector_min = OrientationDetector(confidence_threshold=0.0)
        detector_max = OrientationDetector(confidence_threshold=1.0)

        assert detector_min.confidence_threshold == 0.0
        assert detector_max.confidence_threshold == 1.0

    def test_extreme_time_limits(self):
        """Test extreme time limit values"""
        detector_zero = OrientationDetector(time_limit=0)
        detector_large = OrientationDetector(time_limit=3600)  # 1 hour

        assert detector_zero.time_limit == 0
        assert detector_large.time_limit == 3600

    def test_configuration_persistence(self):
        """Test that configuration persists across method calls"""
        detector = OrientationDetector(confidence_threshold=0.6, time_limit=10)

        # Call various methods and ensure config doesn't change
        detector.reset_stats()
        assert detector.confidence_threshold == 0.6
        assert detector.time_limit == 10

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(detector, 'detect_faces_dnn', return_value=[]):
            with patch.object(detector, 'detect_persons', return_value=[]):
                detector.determine_frame_orientation(frame)
                assert detector.confidence_threshold == 0.6
                assert detector.time_limit == 10

    def test_stats_initialization_with_config(self):
        """Test that stats are properly initialized with different configs"""
        detector1 = OrientationDetector(confidence_threshold=0.4)
        detector2 = OrientationDetector(confidence_threshold=0.9)

        # Both should have same stats structure
        assert 'total_frames' in detector1.stats
        assert 'total_frames' in detector2.stats
        assert detector1.stats['total_frames'] == detector2.stats['total_frames'] == 0

        # But different confidence thresholds
        assert detector1.confidence_threshold == 0.4
        assert detector2.confidence_threshold == 0.9