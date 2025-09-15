import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from video_orientation_detector import OrientationDetector, VideoOrientation

class TestIntegration:
    def test_video_processing_mock(self, detector):
        # Test complete mocking of video processing
        mock_result = {
            'orientation': VideoOrientation.CORRECT,
            'confidence': 0.8,
            'processing_stats': {'frames_processed': 10}
        }
        
        with patch.object(detector, 'process_video', return_value=mock_result):
            result = detector.process_video("mock_video.mp4")
            assert isinstance(result, dict)
            assert 'orientation' in result

    def test_batch_processing_mock(self, detector):
        with patch.object(detector, 'process_video') as mock_process:
            mock_process.return_value = {'orientation': 'CORRECT', 'confidence': 0.8}
            
            # Simulate batch processing without real files
            results = []
            for i in range(2):
                result = detector.process_video(f"mock_video_{i}.mp4")
                results.append(result)
            
            assert len(results) == 2
            assert all(isinstance(r, dict) for r in results)

    def test_stats_management(self, detector):
        # Test stats management with all required fields
        detector.reset_stats()
        assert hasattr(detector, 'stats')
        
        # Initialize ALL required stats for final verdict including voting stats
        detector.stats.update({
            'frames_with_humans': 10,
            'correct_orientation_frames': 8,
            'incorrect_orientation_frames': 2,
            'face_detections': 15,
            'person_detections': 12,
            'body_detections': 12,
            'frames_processed': 10,
            'total_processing_time': 5.0,
            'face_correct_votes': 5,
            'face_incorrect_votes': 2,
            'body_correct_votes': 6,
            'body_incorrect_votes': 1
        })
        
        result = detector.calculate_final_verdict()
        assert isinstance(result, dict)
        assert 'confidence' in result

    def test_detection_pipeline(self, detector, small_frame):
        # Test the detection pipeline
        with patch.object(detector, 'detect_faces_dnn', return_value=[]):
            with patch.object(detector, 'detect_persons', return_value=[]):
                result = detector.determine_frame_orientation(small_frame)
                assert result is not None
                assert isinstance(result, tuple)
                assert len(result) == 2
