import os
import sys

import numpy as np
import pytest

# Keep the project root importable when pytest is launched from another directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import video_orientation_detector as vod  # noqa: E402
from video_orientation_detector import OrientationDetector  # noqa: E402

# The flag lives on the package, not on a local name.
vod.mobilenet_required_override = False


@pytest.fixture
def small_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def detector():
    return OrientationDetector(time_limit=1, confidence_threshold=0.5)
