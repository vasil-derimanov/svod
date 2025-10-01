import json
from video_orientation_detector import OrientationDetector

if __name__ == "__main__":
    detector = OrientationDetector(time_limit=10.0)
    result = detector.process_video_unified(r"C:\Users\boris\Videos\P9080828.mp4", mode="batch", display=False)
    stats = result.detection_info.get("statistics") if hasattr(result, "detection_info") else None
    if stats is None:
        stats = result.get("statistics") if isinstance(result, dict) else {}
    if not stats:
        stats = detector.stats
    rotation_info = {
        "rotation_directions": stats.get("rotation_directions"),
        "rotation_strengths": stats.get("rotation_strengths"),
        "internal_rotation_strengths": getattr(detector, "stats", {}).get(
            "rotation_strengths"
        ),
        "internal_rotation_directions": getattr(detector, "stats", {}).get(
            "rotation_directions"
        ),
    }
    print(json.dumps(rotation_info, indent=2))
