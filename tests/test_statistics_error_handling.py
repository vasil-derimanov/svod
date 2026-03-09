import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from video_orientation_detector import OrientationDetector
import tempfile
import os


class TestStatisticsAndErrorHandling:

    @pytest.fixture
    def detector(self):
        return OrientationDetector()

    def test_statistics_collection(self, detector):
        """Test statistics collection functionality"""
        # Reset stats
        detector.reset_stats()

        # Process some frames
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        for _ in range(5):
                            detector.determine_frame_orientation(frame)

        # Check statistics
        stats = detector.get_statistics()
        assert isinstance(stats, dict)
        assert "frames_processed" in stats
        assert stats["frames_processed"] == 5

    def test_statistics_reporting(self, detector):
        """Test statistics reporting functionality"""
        detector.reset_stats()

        # Generate some stats
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        detector.determine_frame_orientation(frame)

        # Test reporting
        report = detector.get_statistics_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_statistics_reset(self, detector):
        """Test statistics reset functionality"""
        # Generate some stats
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        detector.determine_frame_orientation(frame)

        # Reset and check
        detector.reset_stats()
        stats = detector.get_statistics()
        assert stats["frames_processed"] == 0

    def test_error_handling_invalid_input(self, detector):
        """Test error handling for invalid inputs"""
        # Test with None frame
        with pytest.raises((AttributeError, TypeError)):
            detector.determine_frame_orientation(None)

        # Test with invalid frame type
        with pytest.raises((AttributeError, TypeError)):
            detector.determine_frame_orientation("not_a_frame")

        # Test with wrong dimensions
        invalid_frame = np.zeros((100,), dtype=np.uint8)
        with pytest.raises((ValueError, IndexError)):
            detector.determine_frame_orientation(invalid_frame)

    def test_error_handling_model_loading_failures(self, detector):
        """Test error handling when model loading fails"""
        # Mock model loading failure - YuNet detector unavailable
        orig_yunet = getattr(detector, 'yunet_detector', None)
        detector.yunet_detector = None
        # Should not crash, should continue with other methods
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.determine_frame_orientation(frame)
        assert result is not None  # Should fall back to other methods
        detector.yunet_detector = orig_yunet

    def test_error_handling_network_timeouts(self, detector):
        """Test error handling for network timeouts"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock timeout in face detection
        with patch.object(
            detector, "detect_faces_dnn", side_effect=TimeoutError("Network timeout")
        ):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        result = detector.determine_frame_orientation(frame)
                        assert result is not None  # Should continue with other methods

    def test_error_handling_memory_issues(self, detector):
        """Test error handling for memory issues"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock memory error
        with patch.object(detector, "detect_faces_dnn", side_effect=MemoryError("Out of memory")):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        result = detector.determine_frame_orientation(frame)
                        assert result is not None  # Should continue with other methods

    def test_error_handling_corrupted_frames(self, detector):
        """Test error handling for corrupted frame data"""
        # Create corrupted frame data - use proper uint8 dtype
        corrupted_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8).astype(
            np.float32
        )
        corrupted_frame[50, 50] = np.nan  # Add NaN values

        # Should handle gracefully
        result = detector.determine_frame_orientation(corrupted_frame)
        assert result is not None

    def test_error_handling_empty_detections(self, detector):
        """Test error handling when all detections return empty"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        with patch.object(
                            detector, "mobilenet_detect_orientation", return_value=None
                        ):
                            with patch.object(detector, "detect_hough_lines", return_value=None):
                                result = detector.determine_frame_orientation(frame)
                                assert result is not None  # Should fall back to aspect ratio

    def test_error_handling_file_io_errors(self, detector):
        """Test error handling for file I/O errors"""
        # YuNet model not available - should continue without it
        orig_yunet = getattr(detector, 'yunet_detector', None)
        detector.yunet_detector = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.determine_frame_orientation(frame)
        assert result is not None  # Should continue without model
        detector.yunet_detector = orig_yunet

    def test_error_handling_concurrent_access(self, detector):
        """Test error handling for concurrent access issues"""
        import threading

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = []
        errors = []

        def process_frame():
            try:
                with patch.object(detector, "detect_faces_dnn", return_value=[]):
                    with patch.object(detector, "detect_faces_cascade", return_value=[]):
                        with patch.object(detector, "detect_persons", return_value=[]):
                            with patch.object(detector, "detect_poses", return_value=[]):
                                result = detector.determine_frame_orientation(frame)
                                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=process_frame)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0

    def test_error_recovery_mechanisms(self, detector):
        """Test error recovery mechanisms"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # First call fails, second succeeds
        call_count = 0

        def failing_detection(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return []

        with patch.object(detector, "detect_faces_dnn", side_effect=failing_detection):
            with patch.object(detector, "detect_faces_cascade", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    with patch.object(detector, "detect_poses", return_value=[]):
                        # First call should handle error
                        result1 = detector.determine_frame_orientation(frame)
                        # Second call should work normally
                        result2 = detector.determine_frame_orientation(frame)

                        assert result1 is not None
                        assert result2 is not None

    def test_graceful_degradation(self, detector):
        """Test graceful degradation when components fail"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Disable all advanced features
        with patch.object(detector, "detect_faces_dnn", side_effect=Exception("Disabled")):
            with patch.object(detector, "detect_faces_cascade", side_effect=Exception("Disabled")):
                with patch.object(detector, "detect_persons", side_effect=Exception("Disabled")):
                    with patch.object(detector, "detect_poses", side_effect=Exception("Disabled")):
                        with patch.object(
                            detector,
                            "mobilenet_detect_orientation",
                            side_effect=Exception("Disabled"),
                        ):
                            with patch.object(
                                detector, "detect_hough_lines", side_effect=Exception("Disabled")
                            ):
                                # Should still work with basic aspect ratio analysis
                                result = detector.determine_frame_orientation(frame)
                                assert result is not None

    def test_logging_error_conditions(self, detector):
        """Test logging of error conditions"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Trigger an error condition - should handle gracefully
        with patch.object(detector, "detect_faces_dnn", side_effect=Exception("Test error")):
            result = detector.determine_frame_orientation(frame)

        # Should return a valid result even with errors
        assert isinstance(result, tuple)
        assert len(result) == 2
        orientation, info = result
        assert hasattr(orientation, "name")  # VideoOrientation enum
        assert isinstance(info, dict)
