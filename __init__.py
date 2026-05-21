# SVOD - Smart Video Orientation Detector Package
__version__ = "4.27.0"

# Ensure current directory is in path for imports
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import main classes to make them available at package level
try:
    from video_orientation_detector import OrientationDetector
    from svod_models import VideoOrientation, BatchResult, results_to_json, batch_results_to_json
    from svod_statistics import StatisticsMixin

    __all__ = ["OrientationDetector", "VideoOrientation", "BatchResult", "StatisticsMixin",
               "results_to_json", "batch_results_to_json", "__version__"]
except ImportError:
    # Fallback for package-style import
    from .video_orientation_detector import OrientationDetector
    from .svod_models import VideoOrientation, BatchResult, results_to_json, batch_results_to_json
    from .svod_statistics import StatisticsMixin

    __all__ = ["OrientationDetector", "VideoOrientation", "BatchResult", "StatisticsMixin",
               "results_to_json", "batch_results_to_json", "__version__"]
