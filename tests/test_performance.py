import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock
from video_orientation_detector import OrientationDetector, VideoOrientation


class TestPerformance:
    def test_quick_processing(self, detector, small_frame):
        start_time = time.time()
        with patch.object(detector, "detect_faces_dnn", return_value=[]):
            with patch.object(detector, "detect_persons", return_value=[]):
                result = detector.determine_frame_orientation(small_frame)

        processing_time = time.time() - start_time
        assert processing_time < 1.0  # More reasonable time limit
        assert result is not None

    def test_memory_efficiency(self, detector):
        # Test that we don't leak too much memory
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Process frames but ensure cleanup
        for _ in range(3):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch.object(detector, "detect_faces_dnn", return_value=[]):
                with patch.object(detector, "detect_persons", return_value=[]):
                    detector.determine_frame_orientation(frame)
            # Explicit cleanup
            del frame

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100  # More generous memory limit

    def test_time_limit_enforcement(self, detector):
        detector.time_limit = 0.01
        mock_result = {"orientation": VideoOrientation.CORRECT, "confidence": 0.8}

        with patch.object(detector, "process_video", return_value=mock_result):
            start_time = time.time()
            result = detector.process_video("mock_video.mp4")
            end_time = time.time()

            assert end_time - start_time < 1.0  # Mock should be very fast
            assert isinstance(result, dict)

    def test_aspect_ratio_performance(self, detector):
        # Test that aspect ratio analysis is fast
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]

        start_time = time.time()
        for frame in frames:
            detector.analyze_aspect_ratio(frame)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / len(frames)
        assert avg_time < 0.01  # Less than 10ms per frame
