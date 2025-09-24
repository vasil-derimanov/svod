import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from video_orientation_detector import OrientationDetector, VideoOrientation


class TestUtilityMethods:
    """Test utility methods and statistics"""

    def test_remove_duplicates_no_duplicates(self, detector):
        """Test remove_duplicates with no duplicates"""
        detections = [
            {"box": (10, 10, 50, 50), "confidence": 0.9, "type": "face"},
            {"box": (70, 70, 50, 50), "confidence": 0.8, "type": "face"}
        ]

        result = detector.remove_duplicates(detections)
        assert len(result) == 2
        assert result == detections

    def test_remove_duplicates_with_duplicates(self, detector):
        """Test remove_duplicates removes overlapping detections"""
        detections = [
            {"box": (10, 10, 50, 50), "confidence": 0.9, "type": "face"},
            {"box": (15, 15, 50, 50), "confidence": 0.8, "type": "face"}  # Overlaps with first
        ]

        result = detector.remove_duplicates(detections)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9  # Keeps higher confidence

    def test_calculate_iou_no_overlap(self, detector):
        """Test IoU calculation with no overlap"""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 10, 10)

        iou = detector.calculate_iou(box1, box2)
        assert iou == 0.0

    def test_calculate_iou_full_overlap(self, detector):
        """Test IoU calculation with full overlap"""
        box1 = (0, 0, 10, 10)
        box2 = (0, 0, 10, 10)

        iou = detector.calculate_iou(box1, box2)
        assert iou == 1.0

    def test_calculate_iou_partial_overlap(self, detector):
        """Test IoU calculation with partial overlap"""
        box1 = (0, 0, 20, 20)
        box2 = (10, 10, 20, 20)

        iou = detector.calculate_iou(box1, box2)
        assert iou > 0.0 and iou < 1.0

    def test_reset_stats(self, detector):
        """Test statistics reset"""
        # Modify some stats
        detector.stats["total_frames"] = 100
        detector.stats["face_detections"] = 50

        detector.reset_stats()

        assert detector.stats["total_frames"] == 0
        assert detector.stats["face_detections"] == 0
        assert detector.stats["body_detections"] == 0

    def test_analyze_aspect_ratio_portrait(self, detector):
        """Test aspect ratio analysis for portrait"""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)  # Tall, narrow
        result = detector.analyze_aspect_ratio(frame)
        assert result == "portrait"

    def test_analyze_aspect_ratio_landscape(self, detector):
        """Test aspect ratio analysis for landscape"""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)  # Wide, short
        result = detector.analyze_aspect_ratio(frame)
        assert result == "landscape"

    def test_analyze_aspect_ratio_square(self, detector):
        """Test aspect ratio analysis for square"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)  # Square
        result = detector.analyze_aspect_ratio(frame)
        assert result == "square"

    def test_detect_hough_lines(self, detector):
        """Test Hough line detection"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a vertical line
        frame[:, 50] = 255

        result = detector.detect_hough_lines(frame)
        # Result should be a string
        assert isinstance(result, str)
        assert result in ["portrait", "landscape", "unknown"]

    def test_mobilenet_detect_orientation(self, detector):
        """Test MobileNet orientation detection"""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)  # Portrait frame

        result = detector.mobilenet_detect_orientation(frame)
        # Should return portrait for tall frame
        assert isinstance(result, str)

    def test_determine_frame_orientation_no_humans(self, detector):
        """Test frame orientation determination with no humans detected"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        orientation, detection_info = detector.determine_frame_orientation(frame)
        
        assert orientation == VideoOrientation.UNCERTAIN
        assert detection_info["final_decision"] == "no_human_detected"
        assert len(detection_info["faces"]) == 0
        assert len(detection_info["bodies"]) == 0

    def test_determine_frame_orientation_with_faces(self, detector):
        """Test frame orientation determination with faces"""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)  # Portrait frame
        
        # Mock face detection
        with patch.object(detector, 'detect_faces_dnn', return_value=[
            {"box": (20, 20, 60, 80), "confidence": 0.9, "type": "dnn_face"}
        ]), \
             patch.object(detector, 'detect_faces_cascade', return_value=[]), \
             patch.object(detector, 'detect_persons', return_value=[]), \
             patch.object(detector, 'detect_poses', return_value=[]):
            
            orientation, detection_info = detector.determine_frame_orientation(frame)
            
            assert orientation in [VideoOrientation.CORRECT, VideoOrientation.INCORRECT, VideoOrientation.UNCERTAIN]
            assert len(detection_info["faces"]) > 0
            assert "final_decision" in detection_info

    def test_annotate_frame(self, detector):
        """Test frame annotation"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection_info = {
            "faces": [{"box": (10, 10, 30, 40), "confidence": 0.9}],
            "bodies": [],
            "is_close_up": False,
            "primary_detection": "face"
        }
        
        result = detector.annotate_frame(frame, VideoOrientation.CORRECT, detection_info)
        
        # Should return annotated frame (may be same as input if no drawing)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_analyze_face_orientation_evidence_wide_face(self, detector):
        """Test face orientation evidence for wide face"""
        width, height = 100, 200  # Portrait video
        video_aspect = width / height  # Portrait
        
        evidence = detector._analyze_face_orientation(
            0.7, 50, 20, 40, 60, width, height, video_aspect, 0.0  # Wide face
        )
        
        assert isinstance(evidence, dict)
        assert "counterclockwise" in evidence
        assert evidence["counterclockwise"] > 0

    def test_analyze_body_orientation_evidence_wide_body(self, detector):
        """Test body orientation evidence for wide body"""
        width, height = 100, 200  # Portrait video
        video_aspect = width / height  # Portrait
        
        evidence = detector._analyze_body_orientation(
            0.6, 50, 20, 30, 50, width, height, video_aspect, 0.0  # Wide body
        )
        
        assert isinstance(evidence, dict)
        assert "counterclockwise" in evidence
        assert evidence["counterclockwise"] > 0

    def test_analyze_advanced_edge_orientation(self, detector):
        """Test advanced edge orientation analysis"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add horizontal edges
        frame[25, :] = 255
        frame[75, :] = 255
        
        evidence = detector._analyze_advanced_edge_orientation(frame, 0.5)  # Portrait aspect
        
        assert isinstance(evidence, dict)
        assert "clockwise" in evidence
        assert "counterclockwise" in evidence
        assert "none" in evidence

    def test_analyze_motion_patterns(self, detector):
        """Test motion pattern analysis with frame sequence"""
        frame_sequence = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8)
        ]
        
        evidence = detector._analyze_motion_patterns(frame_sequence, 1.0)
        
        assert isinstance(evidence, dict)
        assert "clockwise" in evidence
        assert "counterclockwise" in evidence
        assert "none" in evidence

    def test_cnn_rotation_classifier(self, detector):
        """Test CNN rotation classifier"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        evidence = detector._cnn_rotation_classifier(frame)
        
        assert isinstance(evidence, dict)
        assert "clockwise" in evidence
        assert "counterclockwise" in evidence
        assert "none" in evidence

    def test_mobilenet_detect_orientation_portrait(self, detector):
        """Test MobileNet orientation detection for portrait frame"""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)  # Portrait frame
        result = detector.mobilenet_detect_orientation(frame)
        assert isinstance(result, str)

    def test_detect_hough_lines_landscape(self, detector):
        """Test Hough line detection for landscape frame"""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)  # Landscape frame
        result = detector.detect_hough_lines(frame)
        assert isinstance(result, str)
        assert result in ["portrait", "landscape", "unknown"]

    def test_load_reference_data_invalid_file(self, detector):
        """Test load_reference_data with invalid file"""
        result = detector.load_reference_data("nonexistent.csv")
        assert result is False

    def test_validate_against_reference(self, detector):
        """Test validate_against_reference method"""
        result = detector.validate_against_reference("test.mp4", VideoOrientation.CORRECT)
        assert isinstance(result, dict)

    def test_get_sampling_ranges_v4_12_0(self, detector):
        """Test get_sampling_ranges_v4_12_0 method"""
        ranges = detector.get_sampling_ranges_v4_12_0(1000, 30.0)
        assert isinstance(ranges, list)
        assert len(ranges) > 0

    def test_should_process_frame_v4_12_0(self, detector):
        """Test should_process_frame_v4_12_0 method"""
        ranges = [(0, 100), (200, 300)]
        result = detector.should_process_frame_v4_12_0(50, ranges)
        assert result is True
        result = detector.should_process_frame_v4_12_0(150, ranges)
        assert result is False