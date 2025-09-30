#!/usr/bin/env python3
"""
Standard Single Video Test (SVOD Official Test #1)
MANDATORY: Use only this script for single video testing

Tests a single video file with current SVOD version and provides detailed results.
Supports time limits, confidence thresholds, and comparison with reference data.

Usage:
    python standard_single_test.py <video_path> [--time-limit SECONDS] [--confidence THRESHOLD]

Examples:
    python standard_single_test.py "video.mp4"
    python standard_single_test.py "video.mp4" --time-limit 30
    python standard_single_test.py "C:\\Users\\boris\\Bad_Examples\\P2170127.mp4" --confidence 0.7
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
from typing import Optional

# Add project root to Python path (go up one level from testing/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_orientation_detector import OrientationDetector


def load_reference_data():
    """Load reference orientation data for validation"""
    reference_data = {}
    reference_file = "../reference_orientations.csv"

    if os.path.exists(reference_file):
        try:
            with open(reference_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row["filename"]
                    reference_data[filename] = {
                        "expected": row["expected_orientation"].lower(),
                        "confidence": row["confidence"].lower(),
                        "notes": row.get("notes", ""),
                    }
        except Exception as e:
            print(f"⚠️ Could not load reference data: {e}")

    return reference_data


def test_single_video(
    video_path: str, time_limit: Optional[float] = None, confidence_threshold: float = 0.5
):
    """Test a single video with SVOD"""
    print("🎯 SVOD Standard Single Video Test")
    print("=" * 60)

    # Validate input
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None

    filename = os.path.basename(video_path)
    print(f"📹 Testing: {filename}")
    print(f"📁 Full path: {video_path}")
    print(f"⏱️ Time limit: {time_limit}s" if time_limit else "⏱️ Time limit: Full video")
    print(f"🎚️ Confidence threshold: {confidence_threshold}")
    print()

    # Load reference data
    reference_data = load_reference_data()
    expected_result = reference_data.get(filename)

    if expected_result:
        print(
            f"📋 Expected: {expected_result['expected'].upper()} ({expected_result['confidence']} confidence)"
        )
        print(f"📝 Notes: {expected_result['notes']}")
        print()

    try:
        # Initialize detector
        print("🔧 Initializing SVOD detector...")
        detector = OrientationDetector(
            confidence_threshold=confidence_threshold, time_limit=time_limit
        )
        print("✅ Detector initialized successfully!")
        print()

        # Process video
        print(f"🎬 Processing video...")
        start_time = time.time()

        results = detector.process_video(video_path, display=False)

        end_time = time.time()
        processing_time = end_time - start_time

        # Extract results
        orientation = results.get("orientation", "UNCERTAIN")
        confidence = results.get("confidence", 0.0)
        rotation_angle = results.get("rotation_angle", "N/A")
        method = results.get("method", "unknown")
        recommendation = results.get("recommendation", "No recommendation")

        # Display results
        print()
        print("🎯 SVOD Test Results")
        print("=" * 40)
        print(f"📊 Result: {orientation}")
        print(f"🎚️ Confidence: {confidence:.3f}")
        print(f"🔄 Rotation needed: {rotation_angle}")
        print(f"🔬 Detection method: {method}")
        print(f"💡 Recommendation: {recommendation}")
        print(f"⏱️ Processing time: {processing_time:.2f}s")

        # Compare with expected
        if expected_result:
            expected = expected_result["expected"]
            match = "✅ MATCH" if orientation.lower() == expected.lower() else "❌ MISMATCH"
            print(f"🎯 vs Expected: {match}")

            if orientation.lower() != expected.lower():
                print(f"   Expected: {expected.upper()}")
                print(f"   Got: {orientation.upper()}")

        # Statistics from video processing
        if "stats" in results:
            stats = results["stats"]
            print()
            print("📈 Detection Statistics:")
            for key, value in stats.items():
                if "count" in key.lower() or "frame" in key.lower():
                    print(f"   {key}: {value}")

        print()
        print("✅ Single video test completed successfully!")

        return {
            "filename": filename,
            "orientation": orientation,
            "confidence": confidence,
            "rotation_angle": rotation_angle,
            "method": method,
            "processing_time": processing_time,
            "expected": expected_result["expected"] if expected_result else None,
            "match": (
                orientation.lower() == expected_result["expected"].lower()
                if expected_result
                else None
            ),
        }

    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="SVOD Standard Single Video Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standard_single_test.py video.mp4
  python standard_single_test.py video.mp4 --time-limit 30
  python standard_single_test.py "C:\\path\\to\\video.mp4" --confidence 0.7
        """,
    )

    parser.add_argument("video_path", help="Path to video file to test")
    parser.add_argument(
        "--time-limit", type=float, help="Time limit in seconds (default: full video)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )

    args = parser.parse_args()

    # Run test
    result = test_single_video(args.video_path, args.time_limit, args.confidence)

    # Exit with appropriate code
    if result is None:
        sys.exit(1)
    elif result.get("match") == False:
        print("⚠️ Result differs from expected - review needed")
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
