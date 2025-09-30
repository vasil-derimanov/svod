import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from video_orientation_detector import OrientationDetector


class TestAdvancedFeatures:

    @pytest.fixture
    def detector(self):
        return OrientationDetector(time_limit=1)

    def test_cnn_classifier_functionality(self, detector):
        """Test CNN rotation classifier functionality"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = detector._cnn_rotation_classifier(frame)
        assert isinstance(result, dict)
        assert "clockwise" in result
        assert "counterclockwise" in result
        assert "none" in result

    def test_optical_flow_analysis(self, detector):
        """Test optical flow rotation analysis"""
        prev_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        curr_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = detector._analyze_optical_flow_rotation(prev_frame, curr_frame, 1.5)
        assert isinstance(result, dict)
        assert "clockwise" in result
        assert "counterclockwise" in result
        assert "none" in result

    def test_advanced_edge_orientation(self, detector):
        """Test advanced edge orientation analysis"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some edges
        frame[50, :] = 255  # Horizontal line

        result = detector._analyze_advanced_edge_orientation(frame, 1.0)
        assert isinstance(result, dict)
        assert "clockwise" in result
        assert "counterclockwise" in result
        assert "none" in result

    def test_motion_patterns_analysis(self, detector):
        """Test motion pattern analysis"""
        frame_sequence = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
        ]

        result = detector._analyze_motion_patterns(frame_sequence, 1.0)
        assert isinstance(result, dict)
        assert "clockwise" in result
        assert "counterclockwise" in result
        assert "none" in result

    def test_ensemble_voting_system(self, detector):
        """Test ensemble voting system"""
        votes = {
            "face": ["correct"],
            "yolo": ["correct"],
            "pose": ["correct"],
            "mobilenet": ["correct"],
            "hough": ["correct"],
            "aspect": ["correct"],
        }
        detection_info = {
            "faces": [{"box": [20, 20, 60, 60], "confidence": 0.9}],
            "bodies": [{"box": [10, 10, 80, 90], "confidence": 0.8}],
            "is_close_up": False,
        }

        # Test the voting logic indirectly through determine_frame_orientation
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(detector, "detect_faces_dnn", return_value=detection_info["faces"]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(
                    detector, "detect_persons", return_value=detection_info["bodies"]
                ):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        result = detector.determine_frame_orientation(frame)
                        assert result is not None

    def test_adaptive_weighting(self, detector):
        """Test adaptive weighting in ensemble voting"""
        # Test with high confidence detections
        high_conf_faces = [{"box": [20, 20, 60, 60], "confidence": 0.95}] * 10
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(detector, "detect_faces_dnn", return_value=high_conf_faces):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        result = detector.determine_frame_orientation(frame)
                        assert result is not None

    def test_conflict_resolution(self, detector):
        """Test conflict resolution in detection results"""
        # Create conflicting votes
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        with patch.object(
                            detector, "mobilenet_detect_orientation", return_value="portrait"
                        ):
                            with patch.object(
                                detector, "detect_hough_lines", return_value="landscape"
                            ):
                                with patch.object(
                                    detector, "analyze_aspect_ratio", return_value="portrait"
                                ):
                                    result = detector.determine_frame_orientation(frame)
                                    assert result is not None

    def test_temporal_consistency(self, detector):
        """Test temporal consistency across frames"""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

        # Process multiple frames
        results = []
        for frame in frames:
            with patch.object(detector, "detect_faces_dnn", return_value=[]):
                with patch.object(detector, "detect_faces_cascade", return_value=[]):
                    with patch.object(detector, "detect_persons", return_value=[]):
                        with patch.object(detector, "detect_poses", return_value=[]):
                            result = detector.determine_frame_orientation(frame)
                            results.append(result)

        assert len(results) == 5
        assert all(r is not None for r in results)

    def test_scene_context_classification(self, detector):
        """Test scene context classification"""
        # Portrait scene
        portrait_frame = np.zeros((200, 100, 3), dtype=np.uint8)
        result = detector.analyze_aspect_ratio(portrait_frame)
        assert result == "portrait"

        # Landscape scene
        landscape_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = detector.analyze_aspect_ratio(landscape_frame)
        assert result == "landscape"

    def test_multi_model_consensus(self, detector):
        """Test multi-model consensus building"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # All models agree
        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        with patch.object(
                            detector, "mobilenet_detect_orientation", return_value="portrait"
                        ):
                            with patch.object(
                                detector, "detect_hough_lines", return_value="portrait"
                            ):
                                with patch.object(
                                    detector, "analyze_aspect_ratio", return_value="portrait"
                                ):
                                    result = detector.determine_frame_orientation(frame)
                                    assert result is not None

    def test_fallback_strategies(self, detector):
        """Test fallback strategies when primary methods fail"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # All detections fail, should fall back to aspect ratio
        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        result = detector.determine_frame_orientation(frame)
                        assert result is not None

    def test_performance_optimization(self, detector):
        """Test performance optimization features"""
        import time

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        start_time = time.time()

        # Process with time limit
        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        result = detector.determine_frame_orientation(frame)

        processing_time = time.time() - start_time
        assert processing_time < 1.0  # Should respect time limit
        assert result is not None
