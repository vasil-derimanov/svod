#!/usr/bin/env python3
"""
Video Orientation Detector Comparison Test
Tests both old and new versions of the detector on the same videos
to compare accuracy and performance.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def run_detector_version(
    script_path: str, video_path: str, args: Optional[List[str]] = None
) -> Dict:
    """
    Run a specific version of the detector on a video file

    Args:
        script_path: Path to the detector script
        video_path: Path to the video file
        args: Additional command line arguments

    Returns:
        Dict with results and timing information
    """
    if args is None:
        args = []

    # Build command
    cmd = [sys.executable, script_path, video_path] + args

    start_time = time.time()

    try:
        # Run the detector
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.path.dirname(script_path),
            encoding="utf-8",
            errors="replace",
        )

        processing_time = time.time() - start_time

        # Parse output to extract key information
        output = result.stdout + result.stderr

        # Extract orientation result
        orientation = "UNKNOWN"
        confidence = 0.0

        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            if "CORRECT" in line and "INCORRECT" not in line:
                orientation = "CORRECT"
            elif "INCORRECT" in line:
                orientation = "INCORRECT"
            elif "UNCERTAIN" in line:
                orientation = "UNCERTAIN"
            elif "Confidence:" in line:
                try:
                    confidence_text = line.split("Confidence:")[1].strip()
                    confidence = float(confidence_text.rstrip("%")) / 100.0
                except:
                    pass

        return {
            "success": result.returncode == 0,
            "orientation": orientation,
            "confidence": confidence,
            "processing_time": processing_time,
            "output": output,
            "error": result.stderr if result.returncode != 0 else None,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "orientation": "TIMEOUT",
            "confidence": 0.0,
            "processing_time": time.time() - start_time,
            "output": "",
            "error": "Process timed out after 5 minutes",
        }
    except Exception as e:
        return {
            "success": False,
            "orientation": "ERROR",
            "confidence": 0.0,
            "processing_time": time.time() - start_time,
            "output": "",
            "error": str(e),
        }


def compare_results(old_result: Dict, new_result: Dict) -> Dict:
    """
    Compare results from old and new detector versions

    Returns:
        Dict with comparison metrics
    """
    comparison = {
        "both_successful": old_result["success"] and new_result["success"],
        "orientation_match": False,
        "confidence_difference": 0.0,
        "time_difference": new_result["processing_time"] - old_result["processing_time"],
        "time_improvement": False,
        "old_result": old_result,
        "new_result": new_result,
    }

    if old_result["success"] and new_result["success"]:
        # Compare orientations
        comparison["orientation_match"] = old_result["orientation"] == new_result["orientation"]

        # Compare confidence
        comparison["confidence_difference"] = new_result["confidence"] - old_result["confidence"]

        # Compare timing
        if new_result["processing_time"] < old_result["processing_time"]:
            comparison["time_improvement"] = True

    return comparison


def run_comparison_test(video_files: List[str], old_script: str, new_script: str) -> Dict:
    """
    Run comparison test on multiple video files

    Args:
        video_files: List of video file paths to test
        old_script: Path to old version of detector
        new_script: Path to new version of detector

    Returns:
        Dict with test results and summary
    """
    results = []
    summary = {
        "total_files": len(video_files),
        "successful_comparisons": 0,
        "orientation_matches": 0,
        "time_improvements": 0,
        "average_time_difference": 0.0,
        "average_confidence_difference": 0.0,
    }

    print("🚀 Starting Video Orientation Detector Comparison Test")
    print("=" * 60)
    print(f"Testing {len(video_files)} video files")
    print(f"Old version: {os.path.basename(old_script)}")
    print(f"New version: {os.path.basename(new_script)}")
    print()

    for i, video_path in enumerate(video_files, 1):
        print(f"📹 Testing video {i}/{len(video_files)}: {os.path.basename(video_path)}")

        # Run old version
        print("  🔄 Running old version...")
        old_result = run_detector_version(old_script, video_path, ["--no-display"])

        # Run new version
        print("  🔄 Running new version...")
        new_result = run_detector_version(new_script, video_path, ["--no-display"])

        # Compare results
        comparison = compare_results(old_result, new_result)

        # Update summary
        if comparison["both_successful"]:
            summary["successful_comparisons"] += 1
            if comparison["orientation_match"]:
                summary["orientation_matches"] += 1
            if comparison["time_improvement"]:
                summary["time_improvements"] += 1

            summary["average_time_difference"] += comparison["time_difference"]
            summary["average_confidence_difference"] += comparison["confidence_difference"]

        # Print individual result
        if comparison["both_successful"]:
            match_icon = "✅" if comparison["orientation_match"] else "❌"
            time_icon = "⚡" if comparison["time_improvement"] else "🐌"

            print(
                f"  {match_icon} Orientation: {old_result['orientation']} → {new_result['orientation']}"
            )
            print(".1f")
            print(".1f")
            print(".1f")
        else:
            print("  ❌ One or both versions failed to process the video")
            if old_result["error"]:
                print(f"    Old version error: {old_result['error']}")
            if new_result["error"]:
                print(f"    New version error: {new_result['error']}")

        results.append({"video_path": video_path, "comparison": comparison})

        print()

    # Calculate averages
    if summary["successful_comparisons"] > 0:
        summary["average_time_difference"] /= summary["successful_comparisons"]
        summary["average_confidence_difference"] /= summary["successful_comparisons"]

    # Print summary
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total videos tested: {summary['total_files']}")
    print(f"Successful comparisons: {summary['successful_comparisons']}")
    print(".1f")
    print(f"Orientation matches: {summary['orientation_matches']}")
    print(f"Time improvements: {summary['time_improvements']}")
    print()

    if summary["successful_comparisons"] > 0:
        print("📈 AVERAGE METRICS")
        print(".1f")
        print(".3f")
        print()

    return {"results": results, "summary": summary}


def save_results_to_file(results: Dict, output_file: str):
    """Save test results to JSON file"""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new video orientation detector")
    parser.add_argument("videos", nargs="+", help="Video files to test")
    parser.add_argument(
        "--old-script",
        default="video_orientation_detector_old.py",
        help="Path to old version script",
    )
    parser.add_argument(
        "--new-script", default="video_orientation_detector.py", help="Path to new version script"
    )
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.old_script):
        print(f"❌ Old script not found: {args.old_script}")
        return 1

    if not os.path.exists(args.new_script):
        print(f"❌ New script not found: {args.new_script}")
        return 1

    # Validate video files
    video_files = []
    for video_path in args.videos:
        if os.path.exists(video_path):
            video_files.append(video_path)
        else:
            print(f"⚠️  Video file not found: {video_path}")

    if not video_files:
        print("❌ No valid video files found")
        return 1

    # Run comparison test
    results = run_comparison_test(video_files, args.old_script, args.new_script)

    # Save results if requested
    if args.output:
        save_results_to_file(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
