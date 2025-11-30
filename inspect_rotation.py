"""Developer helper for inspecting rotation strengths outside the main CLI.

The script is intentionally lightweight so contributors can audit rotation
statistics quickly. It may be folded into the primary CLI once automation
needs cover this workflow.
"""

import argparse
import json
from pathlib import Path

from video_orientation_detector import OrientationDetector


def _extract_rotation_stats(detector: OrientationDetector, result) -> dict:
    stats = result.detection_info.get("statistics") if hasattr(result, "detection_info") else None
    if stats is None:
        stats = result.get("statistics") if isinstance(result, dict) else {}
    if not stats:
        stats = getattr(detector, "stats", {})
    return {
        "rotation_directions": stats.get("rotation_directions"),
        "rotation_strengths": stats.get("rotation_strengths"),
        "internal_rotation_strengths": getattr(detector, "stats", {}).get("rotation_strengths"),
        "internal_rotation_directions": getattr(detector, "stats", {}).get("rotation_directions"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect rotation statistics for a video or batch run."
    )
    parser.add_argument("path", type=Path, help="Video file or directory to analyze")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Max seconds to process before sampling stops (default: 10.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON payload; prints to stdout when omitted.",
    )
    args = parser.parse_args()

    detector = OrientationDetector(time_limit=args.time_limit)
    mode = "batch" if args.path.is_dir() else "single"
    result = detector.process_video_unified(str(args.path), mode=mode, display=False)
    rotation_info = _extract_rotation_stats(detector, result)

    payload = json.dumps(rotation_info, indent=2)
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
