#!/usr/bin/env python3
"""
Detailed test of all Bad_Examples videos to show specific orientation suggestions
"""
import os
import subprocess
import sys
from pathlib import Path


def get_video_files(directory):
    """Get all video files from directory"""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
    video_files = []

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []

    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(directory, file))

    return sorted(video_files)


def test_single_video(video_path):
    """Test a single video and extract key results"""
    try:
        # Run the detector with time limit
        result = subprocess.run(
            [sys.executable, "video_orientation_detector.py", video_path, "--time-limit", "8"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

        # Extract key information from output
        lines = result.stdout.split("\n")
        orientation = "UNKNOWN"
        recommendation = "No recommendation found"
        confidence = "0%"

        for line in lines:
            line = line.strip()
            if "[ERROR] INCORRECT" in line:
                orientation = "INCORRECT"
            elif "[OK] CORRECT" in line:
                orientation = "CORRECT"
            elif "UNCERTAIN" in line:
                orientation = "UNCERTAIN"
            elif "Confidence:" in line:
                confidence = line.split("Confidence:")[1].strip()
            elif "Recommendation:" in line:
                recommendation = line.split("Recommendation:")[1].strip()

        return {
            "success": True,
            "orientation": orientation,
            "recommendation": recommendation,
            "confidence": confidence,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "orientation": "TIMEOUT",
            "recommendation": "Processing timeout",
            "confidence": "0%",
            "error": "Timeout after 120 seconds",
        }
    except Exception as e:
        return {
            "success": False,
            "orientation": "ERROR",
            "recommendation": "Processing failed",
            "confidence": "0%",
            "error": str(e),
        }


def main():
    bad_examples_dir = r"C:\Users\boris\Bad_Examples"

    print("=" * 90)
    print("DETAILED BAD_EXAMPLES ORIENTATION SUGGESTIONS")
    print("=" * 90)
    print()

    video_files = get_video_files(bad_examples_dir)

    if not video_files:
        print("No video files found in Bad_Examples directory!")
        return

    print(f"Testing {len(video_files)} videos from Bad_Examples directory:")
    print("-" * 90)

    successful_tests = 0
    total_tests = len(video_files)

    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        print(f"\n📹 Video {i}/{total_tests}: {filename}")
        print("-" * 50)

        result = test_single_video(video_path)

        if result["success"]:
            successful_tests += 1
            print(f"✅ Status: {result['orientation']}")
            print(f"🎯 Recommendation: {result['recommendation']}")
            print(f"📊 Confidence: {result['confidence']}")
        else:
            print(f"❌ Error: {result['error']}")
            print(f"⚠️  Status: {result['orientation']}")

    print("\n" + "=" * 90)
    print("SUMMARY:")
    print("-" * 20)
    print(f"Total videos: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
