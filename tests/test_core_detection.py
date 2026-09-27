import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_orientation_detector import OrientationDetector, VideoOrientation


class TestCoreDetection:

    @pytest.fixture
    def detector(self):
        return OrientationDetector(time_limit=1)

    @pytest.fixture
    def sample_frame(self):
        return np.zeros((1920, 1080, 3), dtype=np.uint8)

    def test_initialization(self, detector):
        assert detector.time_limit == 1
        assert detector.confidence_threshold == 0.5

    def test_determine_frame_orientation(self, detector):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        result = detector.determine_frame_orientation(frame)
                        assert result is not None

    def test_face_detection(self, detector, sample_frame):
        mock_faces = [{"box": [20, 20, 60, 60], "confidence": 0.9}]

        with patch.object(detector, "detect_faces_dnn", return_value=mock_faces):
            result = detector.detect_faces_dnn(sample_frame)
            assert isinstance(result, list)
            assert len(result) == 1

    def test_duplicate_removal(self, detector):
        detections = [
            {"box": [10, 10, 50, 50], "confidence": 0.9},
            {"box": [12, 12, 48, 48], "confidence": 0.8},
            {"box": [100, 100, 150, 150], "confidence": 0.7},
        ]

        result = detector.remove_duplicates(detections)
        assert len(result) == 2

    def test_iou_calculation(self, detector):
        box1 = [10, 10, 50, 50]
        box2 = [20, 20, 40, 40]

        iou = detector.calculate_iou(box1, box2)
        assert 0 <= iou <= 1

    def test_aspect_ratio_analysis(self, detector):
        portrait = np.zeros((200, 100, 3), dtype=np.uint8)
        result = detector.analyze_aspect_ratio(portrait)
        assert result == "portrait"

        landscape = np.zeros((100, 200, 3), dtype=np.uint8)
        result = detector.analyze_aspect_ratio(landscape)
        assert result == "landscape"

    def test_statistics_tracking(self, detector):
        detector.reset_statistics()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        detector.determine_frame_orientation(frame)

        stats = detector.get_statistics()
        assert isinstance(stats, dict)
        assert "frames_processed" in stats

    def test_error_handling(self, detector):
        with pytest.raises((AttributeError, TypeError)):
            detector.determine_frame_orientation(None)


class TestBodyVoteOverride:
    """Landscape body-vote override in calculate_final_verdict (v4.27.1)."""

    @staticmethod
    def _landscape_stats(detector, **overrides):
        detector.reset_stats()
        detector.video_aspect_ratio = 1920 / 1080
        # Recorded from P7061239.mp4 (reference: incorrect, 90° counterclockwise)
        detector.stats.update(
            {
                "total_frames": 39,
                "frames_with_humans": 32,
                "correct_orientation_frames": 27,
                "incorrect_orientation_frames": 5,
                "face_detections": 19,
                "body_detections": 24,
                "original_body_detections": 23,
                "face_correct_votes": 3,
                "face_incorrect_votes": 0,
                "body_correct_votes": 4,
                "body_incorrect_votes": 10,
                "num_segments": 2,
                "rotation_directions": [
                    "clockwise", "counterclockwise", "none", "clockwise", "counterclockwise"
                ],
            }
        )
        detector.stats.update(overrides)

    def test_p7061239_pattern_is_incorrect(self, detector):
        self._landscape_stats(detector)
        result = detector.calculate_final_verdict()
        assert result["orientation"] == "INCORRECT"
        assert "counterclockwise" in result["recommendation"]

    def test_strong_frame_consensus_stays_correct(self, detector):
        self._landscape_stats(
            detector, correct_orientation_frames=30, incorrect_orientation_frames=2
        )
        assert detector.calculate_final_verdict()["orientation"] == "CORRECT"

    def test_body_majority_correct_stays_correct(self, detector):
        self._landscape_stats(detector, body_correct_votes=10, body_incorrect_votes=4)
        assert detector.calculate_final_verdict()["orientation"] == "CORRECT"
