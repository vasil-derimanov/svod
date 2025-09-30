#!/usr/bin/env python3
"""
Quick test of video orientation detection with configurable video path
"""

import os
import sys
import argparse

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_orientation_detector import OrientationDetector


def test_video_orientation(video_path=None):
    # Define common locations for P2170127.mp4
    video_paths = [
        r"C:\Users\boris\Videos\P2170127.mp4",
        r"C:\Users\boris\Bad_Examples\P2170127.mp4",
        r"C:\Users\boris\Good_Examples\P2170127.mp4",
    ]

    # If no path provided, check for P2170127.mp4 in common locations
    if not video_path:
        video_path = None
        for path in video_paths:
            if os.path.exists(path):
                video_path = path
                break

    if video_path and os.path.exists(video_path):
        print(f"🎬 Found video at: {video_path}")
        detector = OrientationDetector()
        result = detector.process_video_quick(video_path)
        print(f"📊 Orientation: {result.orientation}")
        print(f"🎯 Confidence: {result.confidence:.2%}")
        print(f"⏱️  Processing time: {result.processing_time:.1f}s")

        if hasattr(result, "detection_info") and result.detection_info:
            info = result.detection_info
            if "rotation_direction" in info:
                print(f'🔄 Rotation direction: {info["rotation_direction"]}')

        print(
            f"✅ SUCCESS: Detected as INCORRECT"
            if "INCORRECT" in str(result.orientation)
            else f"❌ ISSUE: Not detected as INCORRECT"
        )
    else:
        print(f"❌ Video file not found: {video_path}")
        if not video_path:
            print("Expected locations checked:")
            for path in video_paths:
                print(f"  • {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test video orientation detection")
    parser.add_argument("--video-path", type=str, help="Path to video file to test")
    args = parser.parse_args()

    test_video_orientation(args.video_path)
