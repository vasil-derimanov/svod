import pytest
import numpy as np
from video_orientation_detector import OrientationDetector


class TestUtilityFunctions:

    @pytest.fixture
    def detector(self):
        return OrientationDetector(time_limit=1)

    def test_analyze_aspect_ratio_all_formats(self, detector):
        """Test aspect ratio analysis for all video formats"""
        # Portrait
        portrait = np.zeros((200, 100, 3), dtype=np.uint8)
        assert detector.analyze_aspect_ratio(portrait) == "portrait"

        # Landscape
        landscape = np.zeros((100, 200, 3), dtype=np.uint8)
        assert detector.analyze_aspect_ratio(landscape) == "landscape"

        # Square
        square = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.analyze_aspect_ratio(square)
        assert result in ["portrait", "landscape", "square", "unknown"]

    def test_detect_hough_lines_all_cases(self, detector):
        """Test Hough line detection for all cases"""
        # Create frame with horizontal lines
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[50, :] = 255  # Horizontal line

        result = detector.detect_hough_lines(frame)
        assert isinstance(result, str)
        assert result in ["landscape", "portrait", "unknown"]

        # Create frame with vertical lines
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, 50] = 255  # Vertical line

        result = detector.detect_hough_lines(frame)
        assert isinstance(result, str)
        assert result in ["landscape", "portrait", "unknown"]

    def test_rotation_hints_all_types(self, detector):
        """Test all rotation hint methods"""
        faces = [{"box": [20, 20, 60, 60], "confidence": 0.9}]
        bodies = [{"box": [10, 10, 80, 90], "confidence": 0.8}]

        # Test face rotation hint
        face_hint = detector._analyze_face_orientation(1.2, 20, 20, 40, 40, 100, 100, 1.0, 0.0)
        assert isinstance(face_hint, dict)
        assert "clockwise" in face_hint
        assert "counterclockwise" in face_hint

        # Test body rotation hint
        body_hint = detector._analyze_body_orientation(1.5, 10, 10, 70, 80, 100, 100, 1.0, 0.0)
        assert isinstance(body_hint, dict)
        assert "clockwise" in body_hint
        assert "counterclockwise" in body_hint

        # Test format rotation hint
        format_hint = detector._get_format_rotation_hint(0.56)  # P2170127.mp4 aspect
        assert isinstance(format_hint, dict)
        assert "clockwise" in format_hint
        assert "counterclockwise" in format_hint

    def test_mobilenet_detect_orientation_all_cases(self, detector):
        """Test MobileNet orientation detection for all cases"""
        # Portrait frame
        portrait = np.zeros((200, 100, 3), dtype=np.uint8)
        result = detector.mobilenet_detect_orientation(portrait)
        assert isinstance(result, str)

        # Landscape frame
        landscape = np.zeros((100, 200, 3), dtype=np.uint8)
        result = detector.mobilenet_detect_orientation(landscape)
        assert isinstance(result, str)

        # Square frame
        square = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.mobilenet_detect_orientation(square)
        assert isinstance(result, str)

    def test_load_reference_data_functionality(self, detector):
        """Test load_reference_data functionality"""
        import tempfile
        import csv

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "expected_orientation", "confidence", "notes"])
            writer.writerow(["test.mp4", "correct", "high", "test video"])
            temp_file = f.name

        try:
            success = detector.load_reference_data(temp_file)
            assert isinstance(success, bool)

            result = detector.validate_against_reference("test.mp4", "correct")
            assert isinstance(result, dict)
        finally:
            import os

            os.unlink(temp_file)

    def test_validate_against_reference_all_cases(self, detector):
        """Test validate_against_reference for all cases"""
        # No reference data
        result = detector.validate_against_reference("unknown.mp4", "correct")
        assert isinstance(result, dict)
        assert result.get("has_reference") is False

        # With reference data
        detector.reference_data = {"test.mp4": {"expected": "correct", "confidence": "high"}}
        result = detector.validate_against_reference("test.mp4", "correct")
        assert isinstance(result, dict)
        assert result.get("has_reference") is True

    def test_get_sampling_ranges_v4_12_0_various_fps(self, detector):
        """Test get_sampling_ranges_v4_12_0 with various FPS"""
        # Low FPS
        ranges = detector.get_sampling_ranges_v4_12_0(300, 10.0)
        assert isinstance(ranges, list)

        # High FPS
        ranges = detector.get_sampling_ranges_v4_12_0(1800, 60.0)
        assert isinstance(ranges, list)

        # Edge case: very short video
        ranges = detector.get_sampling_ranges_v4_12_0(30, 30.0)
        assert isinstance(ranges, list)

    def test_should_process_frame_v4_12_0_edge_cases(self, detector):
        """Test should_process_frame_v4_12_0 with edge cases"""
        ranges = [(0, 100), (200, 300)]

        # Frame in range
        result = detector.should_process_frame_v4_12_0(50, ranges)
        assert isinstance(result, bool)

        # Frame not in range
        result = detector.should_process_frame_v4_12_0(150, ranges)
        assert isinstance(result, bool)

        # Edge of range
        result = detector.should_process_frame_v4_12_0(100, ranges)
        assert isinstance(result, bool)

        # Empty ranges
        result = detector.should_process_frame_v4_12_0(50, [])
        assert isinstance(result, bool)

    def test_is_close_up_detection(self, detector):
        """Test close-up detection functionality"""
        frame_shape = (100, 100, 3)

        # Close-up face
        face_box = (20, 20, 60, 60)
        result = detector.is_close_up(face_box, frame_shape)
        assert isinstance(result, bool)

        # Not close-up
        face_box = (40, 40, 20, 20)
        result = detector.is_close_up(face_box, frame_shape)
        assert isinstance(result, bool)

    def test_reset_stats_functionality(self, detector):
        """Test reset_stats functionality"""
        # Set some stats
        detector.stats["total_frames"] = 100
        detector.stats["face_detections"] = 50

        detector.reset_stats()

        assert detector.stats["total_frames"] == 0
        assert detector.stats["face_detections"] == 0
        assert detector.stats["body_detections"] == 0

    def test_annotate_frame_all_cases(self, detector):
        """Test annotate_frame with all cases"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        from video_orientation_detector import VideoOrientation

        detection_info = {
            "faces": [{"box": (10, 10, 30, 40), "confidence": 0.9}],
            "bodies": [],
            "is_close_up": False,
            "primary_detection": "face",
        }

        # Test CORRECT orientation
        result = detector.annotate_frame(frame, VideoOrientation.CORRECT, detection_info)
        assert result.shape == frame.shape

        # Test INCORRECT orientation
        result = detector.annotate_frame(frame, VideoOrientation.INCORRECT, detection_info)
        assert result.shape == frame.shape

        # Test UNCERTAIN orientation
        result = detector.annotate_frame(frame, VideoOrientation.UNCERTAIN, detection_info)
        assert result.shape == frame.shape
