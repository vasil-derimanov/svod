import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from video_orientation_detector import OrientationDetector


class TestFaceDetection:
    """Test face detection methods"""

    def test_detect_faces_dnn_when_disabled(self, detector):
        """Test detect_faces_dnn returns empty list when DNN is disabled"""
        detector.use_dnn_face = False
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_faces_dnn(frame)
        assert result == []

    def test_detect_faces_dnn_with_mock(self, detector):
        """Test detect_faces_dnn with mocked OpenCV"""
        detector.use_dnn_face = True

        # Mock the face_net
        mock_net = MagicMock()
        # Mock forward to return detections with one face
        # Shape: (1, 1, num_detections, 7) where 7 = [img_id, label, conf, x1, y1, x2, y2]
        mock_net.forward.return_value = np.array([[[[0, 0, 0.8, 0.1, 0.1, 0.9, 0.9]]]])
        detector.face_net = mock_net

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_faces_dnn(frame)

        assert len(result) == 1
        assert result[0]["box"] == (10, 10, 80, 80)
        assert result[0]["confidence"] == 0.8
        assert result[0]["type"] == "dnn_face"

    def test_detect_faces_cascade(self, detector):
        """Test detect_faces_cascade with mocked cascade"""
        # Mock the cascade classifiers
        mock_frontal = MagicMock()
        mock_frontal.detectMultiScale.return_value = [(10, 10, 50, 50)]
        detector.face_cascade = mock_frontal

        mock_profile = MagicMock()
        mock_profile.detectMultiScale.return_value = [(20, 20, 40, 40)]
        detector.profile_cascade = mock_profile

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_faces_cascade(frame)

        assert len(result) == 2
        assert result[0]["box"] == (10, 10, 50, 50)
        assert result[0]["confidence"] == 0.7
        assert result[0]["type"] == "cascade_frontal"
        assert result[1]["box"] == (20, 20, 40, 40)
        assert result[1]["confidence"] == 0.6
        assert result[1]["type"] == "cascade_profile"

    def test_detect_eyes_in_face(self, detector):
        """Test eye detection within face region"""
        # Mock the eye cascade
        mock_eye_cascade = MagicMock()
        mock_eye_cascade.detectMultiScale.return_value = [(10, 15, 8, 6), (25, 15, 8, 6)]

        with patch('cv2.CascadeClassifier', return_value=mock_eye_cascade):
            # Create a face region (grayscale)
            face_region = np.zeros((50, 50), dtype=np.uint8)
            result = detector.detect_eyes_in_face(face_region)

            assert len(result) == 2
            assert result[0] == (10, 15, 8, 6)
            assert result[1] == (25, 15, 8, 6)

    def test_analyze_face_orientation_upright(self, detector):
        """Test face orientation analysis for upright face"""
        # Mock eye detection to return horizontal eyes
        with patch.object(detector, 'detect_eyes_in_face') as mock_detect:
            mock_detect.return_value = [(10, 20, 5, 3), (25, 20, 5, 3)]  # Eyes at same Y level

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            face_box = (5, 5, 40, 50)  # Tall face
            result = detector.analyze_face_orientation(frame, face_box)

            assert result == "upright"

    def test_analyze_face_orientation_sideways(self, detector):
        """Test face orientation analysis for sideways face"""
        # Mock eye detection to return no eyes (fallback to aspect ratio)
        with patch.object(detector, 'detect_eyes_in_face') as mock_detect:
            mock_detect.return_value = []

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            face_box = (5, 5, 50, 30)  # Wide face
            result = detector.analyze_face_orientation(frame, face_box)

            assert result == "sideways"

    def test_analyze_face_orientation_uncertain(self, detector):
        """Test face orientation analysis for uncertain case"""
        # Mock eye detection to return no eyes
        with patch.object(detector, 'detect_eyes_in_face') as mock_detect:
            mock_detect.return_value = []

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            face_box = (5, 5, 40, 40)  # Square face
            result = detector.analyze_face_orientation(frame, face_box)

            assert result == "uncertain"

    def test_analyze_face_orientation_empty_region(self, detector):
        """Test face orientation analysis with empty face region"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        face_box = (50, 50, 60, 60)  # Box outside frame bounds
        result = detector.analyze_face_orientation(frame, face_box)

        assert result == "uncertain"

    def test_is_close_up_true(self, detector):
        """Test close-up detection returns True"""
        face_box = (20, 20, 60, 60)  # Large face relative to frame
        frame_shape = (100, 100, 3)
        result = detector.is_close_up(face_box, frame_shape)

        assert result is True

    def test_is_close_up_false(self, detector):
        """Test close-up detection returns False"""
        face_box = (40, 40, 20, 20)  # Small face relative to frame
        frame_shape = (100, 100, 3)
        result = detector.is_close_up(face_box, frame_shape)

        assert result is False