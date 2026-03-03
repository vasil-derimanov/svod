# SVOD - Smart Video Orientation Detector Package
__version__ = "4.24.0"

# Ensure current directory is in path for imports
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import main classes to make them available at package level
try:
    from video_orientation_detector import OrientationDetector, VideoOrientation, BatchResult

    __all__ = ["OrientationDetector", "VideoOrientation", "BatchResult", "__version__"]
except ImportError:
    # Fallback for package-style import
    from .video_orientation_detector import OrientationDetector, VideoOrientation, BatchResult

    __all__ = ["OrientationDetector", "VideoOrientation", "BatchResult", "__version__"]
