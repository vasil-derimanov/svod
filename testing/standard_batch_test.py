#!/usr/bin/env python3
"""
Standard Batch Test (SVOD Official Test #2)
MANDATORY: Use only this script for batch testing

Tests multiple videos from Good_Examples and Bad_Examples folders.
Compares results with baseline data and reference orientations.

Usage:
    python standard_batch_test.py [--time-limit SECONDS] [--max-files COUNT] [--folder FOLDER]

Examples:
    python standard_batch_test.py
    python standard_batch_test.py --time-limit 30 --max-files 5
    python standard_batch_test.py --folder Good_Examples
"""

import os
import sys
import time
import argparse
import csv
from collections import Counter
from typing import Optional, Dict, List

# Add project root to Python path (go up one level from testing/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_orientation_detector import OrientationDetector


def load_reference_data() -> Dict[str, Dict]:
    """Load reference orientation data for validation"""
    reference_data = {}

    # Resolve reference file path relative to this script location for robustness
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "reference_orientations.csv"),
        os.path.join(script_dir, "reference_orientations.csv"),
        os.path.join(os.path.dirname(script_dir), "reference_orientations.csv"),
    ]

    reference_file = None
    for cand in candidates:
        cand_norm = os.path.normpath(cand)
        if os.path.exists(cand_norm):
            reference_file = cand_norm
            break

    if reference_file is None:
        print("⚠️ reference_orientations.csv not found near testing/; skipping reference accuracy")
        return reference_data

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
        print(f"✅ Loaded {len(reference_data)} reference entries from {reference_file}")
    except Exception as e:
        print(f"⚠️ Could not load reference data from {reference_file}: {e}")

    return reference_data


def get_test_videos(folder_path: str, max_files: Optional[int] = None) -> List[str]:
    """Get video files from test folder"""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return []

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    video_files = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(folder_path, file))

    video_files.sort()  # Consistent order

    if max_files and len(video_files) > max_files:
        video_files = video_files[:max_files]
        print(f"📋 Limited to first {max_files} files")

    return video_files


def test_batch_folder(
    folder_path: str,
    folder_name: str,
    time_limit: Optional[float] = None,
    confidence_threshold: float = 0.5,
    max_files: Optional[int] = None,
) -> Dict:
    """Test all videos in a folder"""
    print(f"\\n🧪 Testing {folder_name} folder")
    print(f"📁 Path: {folder_path}")
    print("=" * 60)

    # Get video files
    video_files = get_test_videos(folder_path, max_files)
    if not video_files:
        return {}

    print(f"📹 Found {len(video_files)} video files to test")

    # Load reference data
    reference_data = load_reference_data()

    # Test results
    results = {}
    total_time = 0

    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        print(f"\\n📹 [{i}/{len(video_files)}] Testing: {filename}")

        try:
            # Create fresh detector for each video (avoid state issues)
            detector = OrientationDetector(
                confidence_threshold=confidence_threshold, time_limit=time_limit
            )

            start_time = time.time()
            video_results = detector.process_video(video_path, display=False)
            end_time = time.time()

            processing_time = end_time - start_time
            total_time += processing_time

            # Extract results
            orientation = video_results.get("orientation", "UNCERTAIN")
            confidence = video_results.get("confidence", 0.0)
            rotation_angle = video_results.get("rotation_angle", "N/A")
            method = video_results.get("method", "unknown")
            recommendation = video_results.get("recommendation", "N/A")

            # Compare with reference
            reference_match = "N/A"
            direction_match = "N/A"
            expected_recommendation = None
            expected = "unknown"
            if filename in reference_data:
                expected = reference_data[filename]["expected"]
                if orientation.lower() == expected.lower():
                    reference_match = "✅ MATCH"
                else:
                    reference_match = f"❌ MISMATCH (expected: {expected})"

                # Compare rotation direction suggestions when expected is incorrect
                notes = reference_data[filename].get("notes", "").lower()
                expected_dir = None
                if expected == "incorrect":
                    if "counterclockwise" in notes:
                        expected_dir = "counterclockwise"
                    elif "clockwise" in notes:
                        expected_dir = "clockwise"
                    # Build canonical expected recommendation if direction known
                    if expected_dir:
                        expected_recommendation = f"rotate 90° {expected_dir}"
                elif expected == "correct":
                    expected_recommendation = "no action needed"

                # Normalize recommendation for comparison
                if isinstance(recommendation, str):
                    rec_norm = recommendation.strip().lower()
                else:
                    rec_norm = str(recommendation).lower()

                if expected_recommendation:
                    if expected_recommendation in rec_norm:
                        direction_match = "✅ DIR MATCH"
                    else:
                        # If we expect a direction, try extracting model's direction
                        model_dir = None
                        if "counterclockwise" in rec_norm:
                            model_dir = "counterclockwise"
                        elif "clockwise" in rec_norm:
                            model_dir = "clockwise"
                        if model_dir and expected_dir and model_dir != expected_dir:
                            direction_match = (
                                f"❌ DIR MISMATCH (expected: {expected_dir}, got: {model_dir})"
                            )
                        else:
                            direction_match = "❌ DIR MISMATCH"

            results[filename] = {
                "orientation": orientation,
                "confidence": confidence,
                "rotation_angle": rotation_angle,
                "method": method,
                "recommendation": recommendation,
                "processing_time": processing_time,
                "expected": expected,
                "expected_recommendation": expected_recommendation,
                "reference_match": reference_match,
                "direction_match": direction_match,
            }

            print(f"   Result: {orientation} (confidence: {confidence:.3f})")
            print(f"   Time: {processing_time:.2f}s")
            print(f"   Reference: {reference_match}")
            print(f"   Recommendation: {recommendation}")
            if direction_match != "N/A":
                print(f"   Direction Check: {direction_match}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[filename] = {
                "orientation": "ERROR",
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": str(e),
            }

    # Summary
    print(f"\\n📊 {folder_name} Summary")
    print("=" * 40)

    if results:
        # Orientation distribution
        orientation_counts = Counter([r.get("orientation", "ERROR") for r in results.values()])
        total_files = len(results)

        print(f"Total files: {total_files}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {total_time/total_files:.2f}s per file")

        print("\\nOrientation Results:")
        for orientation, count in orientation_counts.items():
            percentage = (count / total_files) * 100
            print(f"  {orientation}: {count} files ({percentage:.1f}%)")

        # Reference accuracy
        matches = sum(1 for r in results.values() if r.get("reference_match", "").startswith("✅"))
        total_with_reference = sum(
            1 for r in results.values() if r.get("expected") not in ["unknown", None]
        )

        if total_with_reference > 0:
            accuracy = (matches / total_with_reference) * 100
            print(f"\\nReference Accuracy: {matches}/{total_with_reference} ({accuracy:.1f}%)")

        # Direction suggestion quality (only for files with expected direction available)
        dir_checks = [r for r in results.values() if r.get("expected_recommendation")]
        if dir_checks:
            dir_matches = sum(
                1 for r in dir_checks if str(r.get("direction_match", "")).startswith("✅")
            )
            print(
                f"Direction Suggestion Accuracy: {dir_matches}/{len(dir_checks)} ({(dir_matches/len(dir_checks))*100:.1f}%)"
            )
            # Print mismatches for quick diagnosis
            mismatches = [
                n
                for n, r in results.items()
                if r.get("expected_recommendation")
                and not str(r.get("direction_match", "")).startswith("✅")
            ]
            if mismatches:
                print("  Direction mismatches:")
                for n in mismatches:
                    r = results[n]
                    print(
                        f"   - {n}: expected '{r.get('expected_recommendation')}', got '{str(r.get('recommendation','')).lower()}'"
                    )

    return results


def compare_with_baseline():
    """Show baseline expectations for comparison"""
    print("\\n🏆 Baseline Expectations (v4.21.0)")
    print("=" * 50)
    print("Good_Examples: 90.9% CORRECT (>95% accuracy expected)")
    print("Bad_Examples: 76.9% INCORRECT, 23.1% UNCERTAIN")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="SVOD Standard Batch Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standard_batch_test.py
  python standard_batch_test.py --time-limit 30 --max-files 10
  python standard_batch_test.py --folder Good_Examples
  python standard_batch_test.py --folder Bad_Examples --max-files 5
        """,
    )

    parser.add_argument("--time-limit", type=float, help="Time limit per video in seconds")
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument("--max-files", type=int, help="Maximum files per folder (default: all)")
    parser.add_argument(
        "--folder",
        choices=["Good_Examples", "Bad_Examples"],
        help="Test only specific folder (default: both)",
    )

    args = parser.parse_args()

    print("🚀 SVOD Standard Batch Test")
    print("=" * 60)
    print(
        f"⏱️ Time limit: {args.time_limit}s per video"
        if args.time_limit
        else "⏱️ Time limit: Full videos"
    )
    print(f"🎚️ Confidence threshold: {args.confidence}")
    print(f"📋 Max files: {args.max_files}" if args.max_files else "📋 Max files: All")

    # Test folders
    good_examples_path = r"C:\\Users\\boris\\Good_Examples"
    bad_examples_path = r"C:\\Users\\boris\\Bad_Examples"

    all_results = {}

    if args.folder is None or args.folder == "Good_Examples":
        good_results = test_batch_folder(
            good_examples_path, "Good_Examples", args.time_limit, args.confidence, args.max_files
        )
        all_results["Good_Examples"] = good_results

    if args.folder is None or args.folder == "Bad_Examples":
        bad_results = test_batch_folder(
            bad_examples_path, "Bad_Examples", args.time_limit, args.confidence, args.max_files
        )
        all_results["Bad_Examples"] = bad_results

    # Show baseline comparison
    compare_with_baseline()

    # Final summary
    print(f"\\n🎯 Final Results")
    print("=" * 40)

    total_tested = 0
    total_matches = 0

    for folder_name, results in all_results.items():
        if results:
            tested = len(results)
            matches = sum(
                1 for r in results.values() if r.get("reference_match", "").startswith("✅")
            )
            total_tested += tested
            total_matches += matches

            print(f"{folder_name}: {tested} files tested")
            if any(r.get("expected") not in ["unknown", None] for r in results.values()):
                with_ref = sum(
                    1 for r in results.values() if r.get("expected") not in ["unknown", None]
                )
                if with_ref > 0:
                    accuracy = (matches / with_ref) * 100
                    print(f"  Accuracy: {matches}/{with_ref} ({accuracy:.1f}%)")

    if total_tested > 0:
        overall_accuracy = (total_matches / total_tested) * 100 if total_tested > 0 else 0
        print(f"\\nOverall: {total_matches}/{total_tested} ({overall_accuracy:.1f}%) accuracy")

    print("\\n✅ Batch testing completed!")


if __name__ == "__main__":
    main()
