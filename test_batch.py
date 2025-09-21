#!/usr/bin/env python3
"""
Batch Video Orientation Test Runner
Test multiple videos with both detector versions
"""

import os
import sys
import time
import argparse
from pathlib import Path
import importlib

def find_video_files(directory: str) -> list:
    """Find all video files in directory"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    video_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(os.path.join(root, file))

    return video_files

def test_video_with_detector(detector_script: str, video_path: str) -> tuple:
    """Test a video using direct import instead of subprocess"""
    start_time = time.time()
    try:
        # Dynamically import and reload the detector module
        module_name = Path(detector_script).stem
        if module_name in sys.modules:
            detector_module = importlib.reload(sys.modules[module_name])
        else:
            detector_module = importlib.import_module(module_name)

        # Create detector instance
        detector = detector_module.OrientationDetector(time_limit=30)

        # Process the video
        results = detector.process_video(video_path, display=False)

        processing_time = time.time() - start_time

        # Extract orientation, confidence and recommendation from results
        orientation = results.get('orientation', 'UNCERTAIN')
        confidence = results.get('confidence', 0.0)
        recommendation = results.get('recommendation', 'No recommendation available')

        return True, orientation, confidence, processing_time, None, recommendation

    except Exception as e:
        processing_time = time.time() - start_time
        return False, "ERROR", 0.0, processing_time, str(e), "Error occurred"

def run_batch_test(video_files: list, old_script: str, new_script: str, max_videos: int = 10):
    """Run batch test on multiple videos"""
    print("🎬 Batch Video Orientation Detector Test")
    print("="*60)
    print(f"Testing up to {max_videos} videos from {len(video_files)} found")
    print(f"Old version: {os.path.basename(old_script)}")
    print(f"New version: {os.path.basename(new_script)}")
    print()

    results = []
    tested_count = 0

    for video_path in video_files[:max_videos]:
        tested_count += 1
        print(f"📹 Video {tested_count}/{min(max_videos, len(video_files))}: {os.path.basename(video_path)}")

        # Test old version
        old_success, old_orientation, old_confidence, old_time, old_error, old_recommendation = test_video_with_detector(old_script, video_path)

        # Test new version
        new_success, new_orientation, new_confidence, new_time, new_error, new_recommendation = test_video_with_detector(new_script, video_path)

        # Compare results
        match = old_orientation == new_orientation if old_success and new_success else False
        time_diff = new_time - old_time if old_success and new_success else 0

        # Print result
        success_icon = "✅" if old_success and new_success else "❌"
        match_icon = "✅" if match else "❌" if old_success and new_success else "⚪"
        time_icon = "⚡" if time_diff < 0 and old_success and new_success else "🐌" if time_diff > 0 and old_success and new_success else "⏱️"

        print(f"  {success_icon} {match_icon} {old_orientation} → {new_orientation}")
        if old_success and new_success:
            print(".1f")
            print(f"  📋 New recommendation: {new_recommendation}")
        elif old_error:
            print(f"  ❌ Error: {old_error}")
        print()

        results.append({
            "video": os.path.basename(video_path),
            "old_success": old_success,
            "new_success": new_success,
            "old_orientation": old_orientation,
            "new_orientation": new_orientation,
            "old_recommendation": old_recommendation,
            "new_recommendation": new_recommendation,
            "match": match,
            "old_time": old_time,
            "new_time": new_time,
            "time_diff": time_diff,
            "old_error": old_error,
            "new_error": new_error
        })

    # Print summary
    print("📊 BATCH TEST SUMMARY")
    print("="*60)

    total_tested = len(results)
    successful_tests = sum(1 for r in results if r['old_success'] and r['new_success'])
    matches = sum(1 for r in results if r['match'])
    time_improvements = sum(1 for r in results if r['time_diff'] < 0 and r['old_success'] and r['new_success'])

    print(f"Total videos tested: {total_tested}")
    print(f"Successful comparisons: {successful_tests}")
    print(".1f")
    print(f"Orientation matches: {matches}")
    print(f"Time improvements: {time_improvements}")

    if successful_tests > 0:
        avg_time_diff = sum(r['time_diff'] for r in results if r['old_success'] and r['new_success']) / successful_tests
        print(".1f")

    return results

def main():
    parser = argparse.ArgumentParser(description="Batch test video orientation detectors")
    parser.add_argument("directory", help="Directory containing video files")
    parser.add_argument("--old-script", default="video_orientation_detector_old.py",
                       help="Path to old version script")
    parser.add_argument("--new-script", default="video_orientation_detector.py",
                       help="Path to new version script")
    parser.add_argument("--max-videos", type=int, default=5,
                       help="Maximum number of videos to test")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.directory):
        print(f"❌ Directory not found: {args.directory}")
        return 1

    if not os.path.exists(args.old_script):
        print(f"❌ Old script not found: {args.old_script}")
        return 1

    if not os.path.exists(args.new_script):
        print(f"❌ New script not found: {args.new_script}")
        return 1

    # Find video files
    video_files = find_video_files(args.directory)

    if not video_files:
        print(f"❌ No video files found in {args.directory}")
        return 1

    # Run batch test
    results = run_batch_test(video_files, args.old_script, args.new_script, args.max_videos)

    return 0

if __name__ == "__main__":
    sys.exit(main())