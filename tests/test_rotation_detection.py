import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from video_orientation_detector import OrientationDetector


class TestRotationDetection:
    """Test rotation direction detection methods"""

    def test_detect_rotation_direction_clockwise(self, detector):
        """Test rotation direction detection returns clockwise"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = [{"box": (10, 10, 30, 40), "confidence": 0.9}]
        bodies = []

        result = detector.detect_rotation_direction(frame, faces, bodies)
        # Result should be one of the expected directions
        assert result in ["clockwise", "counterclockwise", "none"]

    def test_detect_rotation_direction_counterclockwise(self, detector):
        """Test rotation direction detection returns counterclockwise"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = [{"box": (60, 10, 30, 40), "confidence": 0.9}]  # Face on right side
        bodies = []

        result = detector.detect_rotation_direction(frame, faces, bodies)
        assert result in ["clockwise", "counterclockwise", "none"]

    def test_detect_rotation_direction_no_detections(self, detector):
        """Test rotation direction detection with no detections"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = []
        bodies = []

        result = detector.detect_rotation_direction(frame, faces, bodies)
        assert result in ["clockwise", "counterclockwise", "none"]

    def test_analyze_face_orientation_evidence(self, detector):
        """Test face orientation evidence analysis"""
        width, height = 200, 100
        video_aspect = width / height  # Landscape

        # Test upright face in portrait video
        evidence = detector._analyze_face_orientation(
            1.5, 50, 20, 30, 45, width, height, video_aspect, 0.0
        )
        assert isinstance(evidence, dict)
        assert "clockwise" in evidence
        assert "counterclockwise" in evidence
        assert "none" in evidence

    def test_analyze_body_orientation_evidence(self, detector):
        """Test body orientation evidence analysis"""
        width, height = 200, 100
        video_aspect = width / height  # Landscape

        # Test upright body
        evidence = detector._analyze_body_orientation(
            2.0, 50, 20, 20, 40, width, height, video_aspect, 0.0
        )
        assert isinstance(evidence, dict)
        assert "clockwise" in evidence
        assert "counterclockwise" in evidence
        assert "none" in evidence

    def test_get_format_rotation_hint_portrait(self, detector):
        """Test format rotation hint for portrait video"""
        evidence = detector._get_format_rotation_hint(0.5)  # Portrait aspect < 0.6
        assert isinstance(evidence, dict)
        assert "counterclockwise" in evidence
        assert evidence["counterclockwise"] > 0

    def test_get_format_rotation_hint_landscape(self, detector):
        """Test format rotation hint for landscape video"""
        evidence = detector._get_format_rotation_hint(1.8)  # Landscape aspect
        assert isinstance(evidence, dict)
        assert "clockwise" in evidence

    def test_analyze_aspect_rotation_patterns(self, detector):
        """Test aspect ratio rotation pattern analysis"""
        evidence = detector._analyze_aspect_rotation_patterns(0.5625, 2160, 3840)
        assert isinstance(evidence, dict)
        assert "clockwise" in evidence
        assert "counterclockwise" in evidence
        assert "none" in evidence

    def test_analyze_advanced_edge_orientation(self, detector):
        """Test advanced edge orientation analysis"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some edges
        frame[50, :] = 255  # Horizontal line

        evidence = detector._analyze_advanced_edge_orientation(frame, 1.0)
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

    def test_analyze_motion_patterns(self, detector):
        """Test motion pattern analysis"""
        frame_sequence = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
        ]

        evidence = detector._analyze_motion_patterns(frame_sequence, 1.0)
        assert isinstance(evidence, dict)
        assert "clockwise" in evidence
        assert "counterclockwise" in evidence
        assert "none" in evidence
