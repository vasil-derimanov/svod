"""Mutable runtime flags shared by the detector and the CLI.

These names used to live on ``video_orientation_detector``. The facade still
exposes them as attributes so ``vod.YOLO_AVAILABLE = True`` keeps working.
"""

YOLO_AVAILABLE = False
mobilenet_required_override = False
TQDM_AVAILABLE = False
