import pytest
import numpy as np
import sys
import os

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_orientation_detector import OrientationDetector, mobilenet_required_override

# Disable MobileNet requirement for tests
mobilenet_required_override = False

@pytest.fixture
def small_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def detector():
    return OrientationDetector(time_limit=1, confidence_threshold=0.5)
