import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from video_orientation_detector import OrientationDetector


class TestPoseDetection:
    """Test pose detection methods"""

    def test_detect_poses_disabled(self, detector):
        """Test detect_poses returns empty list when MediaPipe is disabled"""
        detector.mediapipe_available = False
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_poses(frame)
        assert result == []

    def test_detect_poses_with_mock(self, detector):
        """Test detect_poses with mocked MediaPipe"""
        detector.mediapipe_available = True

        # Mock MediaPipe pose
        mock_pose = MagicMock()
        mock_results = MagicMock()
        mock_landmarks = MagicMock()

        # Mock landmark data
        mock_nose = MagicMock()
        mock_nose.x, mock_nose.y, mock_nose.visibility = 0.5, 0.2, 0.9

        mock_left_shoulder = MagicMock()
        mock_left_shoulder.x, mock_left_shoulder.y, mock_left_shoulder.visibility = 0.4, 0.3, 0.9

        mock_right_shoulder = MagicMock()
        mock_right_shoulder.x, mock_right_shoulder.y, mock_right_shoulder.visibility = 0.6, 0.3, 0.9

        mock_left_hip = MagicMock()
        mock_left_hip.x, mock_left_hip.y, mock_left_hip.visibility = 0.4, 0.7, 0.9

        mock_right_hip = MagicMock()
        mock_right_hip.x, mock_right_hip.y, mock_right_hip.visibility = 0.6, 0.7, 0.9

        mock_landmarks.landmark = [
            mock_nose,
            mock_left_shoulder,
            mock_right_shoulder,
            mock_left_hip,
            mock_right_hip,
        ]
        mock_results.pose_landmarks = mock_landmarks
        mock_pose.process.return_value = mock_results

        detector.mp_pose = MagicMock()
        detector.mp_drawing = MagicMock()
        detector.pose = mock_pose

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.detect_poses(frame)

        assert len(result) == 1
        assert "box" in result[0]
        assert "confidence" in result[0]
        assert "type" in result[0]
        assert result[0]["type"] == "mediapipe_pose"
        assert "landmarks" in result[0]

    def test_detect_poses_no_landmarks(self, detector):
        """Test detect_poses when no landmarks are detected"""
        detector.mediapipe_available = True

        # Mock MediaPipe pose with no landmarks
        mock_pose = MagicMock()
        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        mock_pose.process.return_value = mock_results

        detector.pose = mock_pose

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_poses(frame)

        assert result == []

    def test_detect_poses_exception(self, detector):
        """Test detect_poses handles exceptions gracefully"""
        detector.mediapipe_available = True

        # Mock MediaPipe pose to raise exception
        mock_pose = MagicMock()
        mock_pose.process.side_effect = Exception("MediaPipe error")
        detector.pose = mock_pose

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_poses(frame)

        assert result == []  # Should return empty list on error

    def test_analyze_pose_orientation_upright(self, detector):
        """Test pose orientation analysis for upright pose"""
        pose = {
            "landmarks": {
                "nose": (0.5, 0.2, 0.9),
                "left_shoulder": (0.4, 0.3, 0.9),
                "right_shoulder": (0.6, 0.3, 0.9),
                "left_hip": (0.4, 0.7, 0.9),
                "right_hip": (0.6, 0.7, 0.9),
            }
        }
        result = detector.analyze_pose_orientation(pose)
        assert result == "upright"

    def test_analyze_pose_orientation_rotated(self, detector):
        """Test pose orientation analysis for rotated pose"""
        pose = {
            "landmarks": {
                "nose": (0.2, 0.5, 0.9),  # Nose moved to indicate rotation
                "left_shoulder": (0.3, 0.5, 0.9),
                "right_shoulder": (0.7, 0.5, 0.9),  # Shoulders at same height as hips
                "left_hip": (0.3, 0.5, 0.9),
                "right_hip": (0.7, 0.5, 0.9),  # Hips at same height as shoulders
            }
        }
        result = detector.analyze_pose_orientation(pose)
        assert result == "rotated"

    def test_analyze_pose_orientation_uncertain(self, detector):
        """Test pose orientation analysis for uncertain case"""
        pose = {}  # Empty pose
        result = detector.analyze_pose_orientation(pose)
        assert result == "uncertain"

    def test_analyze_pose_orientation_low_visibility(self, detector):
        """Test pose orientation analysis with low visibility landmarks"""
        pose = {
            "landmarks": {
                "nose": (0.5, 0.2, 0.3),  # Low visibility
                "left_shoulder": (0.4, 0.3, 0.3),
                "right_shoulder": (0.6, 0.3, 0.3),
                "left_hip": (0.4, 0.7, 0.3),
                "right_hip": (0.6, 0.7, 0.3),
            }
        }
        result = detector.analyze_pose_orientation(pose)
        assert result == "uncertain"
