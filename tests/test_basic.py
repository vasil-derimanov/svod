import pytest
import numpy as np
from unittest.mock import patch
from video_orientation_detector import OrientationDetector, VideoOrientation

class TestBasicFunctionality:
    def test_detector_initialization(self, detector):
        assert detector.confidence_threshold == 0.5
        assert detector.time_limit == 1

    def test_aspect_ratio_detection(self, detector):
        portrait = np.zeros((200, 100, 3), dtype=np.uint8)
        landscape = np.zeros((100, 200, 3), dtype=np.uint8)
        assert detector.analyze_aspect_ratio(portrait) == "portrait"
        assert detector.analyze_aspect_ratio(landscape) == "landscape"

    def test_empty_frame_processing(self, detector, small_frame):
        with patch.object(detector, 'detect_faces_dnn', return_value=[]):
            with patch.object(detector, 'detect_persons', return_value=[]):
                result = detector.determine_frame_orientation(small_frame)
                assert result is not None
                assert isinstance(result, tuple)

    def test_face_detection_simple(self, detector, small_frame):
        # Simply test that the method doesn't crash
        result = detector.detect_faces_dnn(small_frame)
        assert isinstance(result, list)

    def test_person_detection_simple(self, detector, small_frame):
        # Simply test that the method doesn't crash
        result = detector.detect_persons(small_frame)
        assert isinstance(result, list)

    def test_rotation_direction_detection(self, detector, small_frame):
        faces = [{'box': [20, 20, 60, 60], 'confidence': 0.9}]
        bodies = [{'box': [10, 10, 80, 90], 'confidence': 0.8}]
        result = detector.detect_rotation_direction(small_frame, faces, bodies)
        assert isinstance(result, str)
        assert result in ['clockwise', 'counterclockwise', 'none']
