import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from video_orientation_detector import OrientationDetector


class TestPersonDetection:
    """Test person/body detection methods"""

    def test_detect_persons_yolov8_disabled(self, detector):
        """Test detect_persons returns empty list when YOLOv8 is disabled"""
        detector.use_yolov8 = False
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_persons(frame)
        assert result == []

    def test_detect_persons_yolov8_with_mock(self, detector):
        """Test detect_persons with mocked YOLOv8"""
        detector.use_yolov8 = True

        # Mock YOLOv8 model
        mock_model = MagicMock()
        mock_result = MagicMock()

        # Create a mock box
        mock_box = MagicMock()
        mock_box.cls = [0]  # Person class
        mock_box.conf = [0.85]
        mock_box.xyxy = [np.array([10, 10, 60, 110])]  # Box coordinates as numpy array

        mock_boxes = [mock_box]  # Make it iterable
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        detector.yolov8_model = mock_model

        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        result = detector.detect_persons(frame)

        assert len(result) == 1
        assert result[0]["box"] == (10, 10, 50, 100)
        assert result[0]["confidence"] == 0.85
        assert result[0]["type"] == "yolov8_person"

    def test_detect_persons_yolov8_exception(self, detector):
        """Test detect_persons handles YOLOv8 exceptions gracefully"""
        detector.use_yolov8 = True

        # Mock YOLOv8 model to raise exception
        mock_model = MagicMock()
        mock_model.side_effect = Exception("YOLOv8 error")
        detector.yolov8_model = mock_model

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_persons(frame)

        assert result == []  # Should return empty list on error

    def test_detect_persons_non_person_class(self, detector):
        """Test detect_persons filters out non-person detections"""
        detector.use_yolov8 = True

        # Mock YOLOv8 model with non-person class
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.cls = [1]  # Non-person class (car, etc.)
        mock_boxes.conf = [0.9]
        mock_boxes.xyxy = [[10, 10, 60, 110]]
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        detector.yolov8_model = mock_model

        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        result = detector.detect_persons(frame)

        assert result == []  # Should filter out non-person detections

    def test_detect_persons_low_confidence(self, detector):
        """Test detect_persons filters out low confidence detections"""
        detector.use_yolov8 = True

        # Mock YOLOv8 model with low confidence
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.cls = [0]  # Person class
        mock_boxes.conf = [0.3]  # Below threshold
        mock_boxes.xyxy = [[10, 10, 60, 110]]
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        detector.yolov8_model = mock_model

        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        result = detector.detect_persons(frame)

        assert result == []  # Should filter out low confidence detections
