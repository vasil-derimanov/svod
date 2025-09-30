#!/usr/bin/env python3
"""
Standard Performance Test (SVOD Official Test #3)
MANDATORY: Use only this script for performance testing

Measures SVOD performance metrics: speed, memory usage, detection rates.
Creates performance baselines and compares with historical data.

Usage:
    python standard_performance_test.py [--test-video PATH] [--iterations COUNT]

Examples:
    python standard_performance_test.py
    python standard_performance_test.py --test-video "video.mp4"
    python standard_performance_test.py --iterations 5
"""

import os
import sys
import time
import argparse
import psutil
import gc
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to Python path (go up one level from testing/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_orientation_detector import OrientationDetector


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_performance_single_video(video_path: str, iterations: int = 3) -> List[Dict]:
    """Test performance on a single video with multiple runs"""
    print(f"🏃 Performance Test: {os.path.basename(video_path)}")
    print(f"📹 Video: {video_path}")
    print(f"🔄 Iterations: {iterations}")
    print("=" * 50)

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return []

    results = []

    for i in range(iterations):
        print(f"\\n🧪 Iteration {i+1}/{iterations}")

        # Force garbage collection
        gc.collect()

        # Measure initial memory
        memory_start = get_memory_usage()

        try:
            # Initialize detector
            start_time = time.time()
            detector = OrientationDetector(confidence_threshold=0.5, time_limit=30.0)
            init_time = time.time() - start_time

            memory_after_init = get_memory_usage()

            # Process video
            process_start = time.time()
            video_results = detector.process_video(video_path, display=False)
            process_end = time.time()

            memory_peak = get_memory_usage()

            processing_time = process_end - process_start

            # Extract metrics
            orientation = video_results.get("orientation", "UNCERTAIN")
            confidence = video_results.get("confidence", 0.0)

            # Get detection statistics
            stats = video_results.get("stats", {})
            face_count = stats.get("face_detections", 0)
            body_count = stats.get("body_detections", 0)
            frames_analyzed = stats.get("frames_analyzed", 0)

            iteration_result = {
                "iteration": i + 1,
                "initialization_time": init_time,
                "processing_time": processing_time,
                "total_time": init_time + processing_time,
                "memory_start": memory_start,
                "memory_after_init": memory_after_init,
                "memory_peak": memory_peak,
                "memory_used": memory_peak - memory_start,
                "orientation": orientation,
                "confidence": confidence,
                "face_detections": face_count,
                "body_detections": body_count,
                "frames_analyzed": frames_analyzed,
                "fps": frames_analyzed / processing_time if processing_time > 0 else 0,
            }

            results.append(iteration_result)

            print(f"   ⏱️ Processing: {processing_time:.2f}s")
            print(f"   🧠 Memory: {memory_peak:.1f}MB (peak)")
            print(f"   📊 Result: {orientation} ({confidence:.3f})")
            print(f"   👤 Faces: {face_count}, Bodies: {body_count}")
            print(
                f"   🎬 FPS: {frames_analyzed / processing_time:.1f}"
                if processing_time > 0
                else "   🎬 FPS: N/A"
            )

        except Exception as e:
            print(f"   ❌ Error in iteration {i+1}: {e}")
            results.append({"iteration": i + 1, "error": str(e)})

    return results


def calculate_performance_stats(results: List[Dict]) -> Dict:
    """Calculate performance statistics from multiple runs"""
    if not results:
        return {}

    # Filter out error results
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return {"error": "No valid results"}

    # Calculate averages
    avg_stats = {
        "iterations": len(valid_results),
        "avg_initialization_time": sum(r["initialization_time"] for r in valid_results)
        / len(valid_results),
        "avg_processing_time": sum(r["processing_time"] for r in valid_results)
        / len(valid_results),
        "avg_total_time": sum(r["total_time"] for r in valid_results) / len(valid_results),
        "avg_memory_used": sum(r["memory_used"] for r in valid_results) / len(valid_results),
        "avg_memory_peak": sum(r["memory_peak"] for r in valid_results) / len(valid_results),
        "avg_fps": sum(r["fps"] for r in valid_results) / len(valid_results),
        "avg_face_detections": sum(r["face_detections"] for r in valid_results)
        / len(valid_results),
        "avg_body_detections": sum(r["body_detections"] for r in valid_results)
        / len(valid_results),
        "consistency_score": calculate_consistency_score(valid_results),
    }

    # Min/Max values
    if len(valid_results) > 1:
        avg_stats.update(
            {
                "min_processing_time": min(r["processing_time"] for r in valid_results),
                "max_processing_time": max(r["processing_time"] for r in valid_results),
                "processing_time_variance": max(r["processing_time"] for r in valid_results)
                - min(r["processing_time"] for r in valid_results),
            }
        )

    return avg_stats


def calculate_consistency_score(results: List[Dict]) -> float:
    """Calculate consistency score based on result variations"""
    if len(results) < 2:
        return 1.0

    orientations = [r["orientation"] for r in results]
    confidences = [r["confidence"] for r in results]

    # Orientation consistency (1.0 if all same, lower if different)
    orientation_consistency = orientations.count(orientations[0]) / len(orientations)

    # Confidence consistency (1.0 if low variance, lower if high variance)
    if len(confidences) > 1:
        confidence_variance = max(confidences) - min(confidences)
        confidence_consistency = max(0.0, 1.0 - confidence_variance)
    else:
        confidence_consistency = 1.0

    return (orientation_consistency + confidence_consistency) / 2.0


def save_performance_baseline(stats: Dict, video_name: str):
    """Save performance data as baseline"""
    if "error" in stats:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_file = f"performance_baselines/performance_v4_22_0_{timestamp}.txt"

    # Create directory if needed
    os.makedirs("performance_baselines", exist_ok=True)

    try:
        with open(baseline_file, "w", encoding="utf-8") as f:
            f.write(f"SVOD Performance Baseline - v4.22.0 (YOLOv10)\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Test Video: {video_name}\\n")
            f.write("=" * 50 + "\\n\\n")

            f.write("Performance Metrics:\\n")
            f.write(f"  Average Processing Time: {stats['avg_processing_time']:.3f}s\\n")
            f.write(f"  Average Initialization Time: {stats['avg_initialization_time']:.3f}s\\n")
            f.write(f"  Average Total Time: {stats['avg_total_time']:.3f}s\\n")
            f.write(f"  Average FPS: {stats['avg_fps']:.1f}\\n")
            f.write(f"  Average Memory Usage: {stats['avg_memory_used']:.1f}MB\\n")
            f.write(f"  Peak Memory: {stats['avg_memory_peak']:.1f}MB\\n")
            f.write(f"  Consistency Score: {stats['consistency_score']:.3f}\\n")

            f.write("\\nDetection Metrics:\\n")
            f.write(f"  Average Face Detections: {stats['avg_face_detections']:.1f}\\n")
            f.write(f"  Average Body Detections: {stats['avg_body_detections']:.1f}\\n")

            if "processing_time_variance" in stats:
                f.write(f"\\nVariability:\\n")
                f.write(f"  Min Processing Time: {stats['min_processing_time']:.3f}s\\n")
                f.write(f"  Max Processing Time: {stats['max_processing_time']:.3f}s\\n")
                f.write(f"  Time Variance: {stats['processing_time_variance']:.3f}s\\n")

        print(f"✅ Baseline saved: {baseline_file}")

    except Exception as e:
        print(f"⚠️ Could not save baseline: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="SVOD Standard Performance Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test-video",
        default=r"C:\\Users\\boris\\Bad_Examples\\20150911_221520.mp4",
        help="Video file for performance testing",
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of test iterations (default: 3)"
    )
    parser.add_argument(
        "--save-baseline", action="store_true", help="Save results as performance baseline"
    )

    args = parser.parse_args()

    print("🚀 SVOD Standard Performance Test")
    print("=" * 60)
    print(f"Version: YOLOv10 v4.22.0")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")

    # Run performance test
    results = test_performance_single_video(args.test_video, args.iterations)

    if not results:
        print("❌ Performance test failed")
        sys.exit(1)

    # Calculate statistics
    stats = calculate_performance_stats(results)

    if "error" in stats:
        print(f"❌ Performance calculation failed: {stats['error']}")
        sys.exit(1)

    # Display results
    print(f"\\n📊 Performance Results")
    print("=" * 50)
    print(f"🔄 Valid iterations: {stats['iterations']}")
    print(f"⏱️ Avg processing time: {stats['avg_processing_time']:.3f}s")
    print(f"🚀 Avg initialization: {stats['avg_initialization_time']:.3f}s")
    print(f"⚡ Average FPS: {stats['avg_fps']:.1f}")
    print(f"🧠 Avg memory usage: {stats['avg_memory_used']:.1f}MB")
    print(f"📈 Peak memory: {stats['avg_memory_peak']:.1f}MB")
    print(f"🎯 Consistency score: {stats['consistency_score']:.3f}")

    print(f"\\n👁️ Detection Performance:")
    print(f"  Faces per run: {stats['avg_face_detections']:.1f}")
    print(f"  Bodies per run: {stats['avg_body_detections']:.1f}")

    if "processing_time_variance" in stats:
        print(f"\\n📊 Time Variability:")
        print(f"  Fastest: {stats['min_processing_time']:.3f}s")
        print(f"  Slowest: {stats['max_processing_time']:.3f}s")
        print(f"  Variance: {stats['processing_time_variance']:.3f}s")

    # Save baseline if requested
    if args.save_baseline:
        save_performance_baseline(stats, os.path.basename(args.test_video))

    print("\\n✅ Performance test completed!")


if __name__ == "__main__":
    main()
