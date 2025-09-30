#!/usr/bin/env python3
"""
Simple Video Orientation Test Runner
Run both old and new detector versions on a single video for quick comparison
"""

import os
import sys
import time
import subprocess
import argparse


def run_single_test(script_path: str, video_path: str, version_name: str) -> dict:
    """Run a single detector version on a video"""
    print(f"\n🔄 Testing with {version_name}...")
    print(f"Script: {os.path.basename(script_path)}")

    start_time = time.time()

    try:
        # Run the detector
        cmd = [sys.executable, script_path, video_path, "--no-display"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(script_path),
            encoding="utf-8",
            errors="replace",
        )

        processing_time = time.time() - start_time

        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print(f"Exit code: {result.returncode}")

        if result.returncode == 0:
            print("✅ Success")
        else:
            print("❌ Failed")

        # Print key output lines
        output_lines = result.stdout.split("\n")
        key_lines = []
        for line in output_lines:
            line = line.strip()
            if any(
                keyword in line.upper()
                for keyword in ["CORRECT", "INCORRECT", "UNCERTAIN", "CONFIDENCE", "VERDICT"]
            ):
                key_lines.append(line)

        if key_lines:
            print("\n📋 Key Results:")
            for line in key_lines[:5]:  # Show first 5 key lines
                print(f"   {line}")

        return {
            "success": result.returncode == 0,
            "processing_time": processing_time,
            "output": result.stdout,
            "error": result.stderr,
            "key_lines": key_lines,
        }

    except subprocess.TimeoutExpired:
        processing_time = time.time() - start_time
        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print("❌ Timeout (5 minutes)")
        return {
            "success": False,
            "processing_time": processing_time,
            "output": "",
            "error": "Timeout after 5 minutes",
            "key_lines": [],
        }
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print(f"❌ Error: {e}")
        return {
            "success": False,
            "processing_time": processing_time,
            "output": "",
            "error": str(e),
            "key_lines": [],
        }


def compare_outputs(old_result: dict, new_result: dict):
    """Compare outputs from both versions"""
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)

    print(f"Old version success: {'✅' if old_result['success'] else '❌'}")
    print(f"New version success: {'✅' if new_result['success'] else '❌'}")

    if old_result["success"] and new_result["success"]:
        time_diff = new_result["processing_time"] - old_result["processing_time"]
        time_icon = "⚡" if time_diff < 0 else "🐌"
        print(f"{time_icon} Time difference: {time_diff:+.1f}s")

        # Compare key results
        old_key = " ".join(old_result["key_lines"][:3])
        new_key = " ".join(new_result["key_lines"][:3])

        if old_key and new_key:
            match = any(
                word in new_key.upper()
                for word in ["CORRECT", "INCORRECT", "UNCERTAIN"]
                if word in old_key.upper()
            )
            result_icon = "✅" if match else "❌"
            print(f"{result_icon} Result consistency: {'Similar' if match else 'Different'}")
    else:
        print("❌ Cannot compare - one or both versions failed")


def main():
    parser = argparse.ArgumentParser(description="Test video orientation detector versions")
    parser.add_argument("video", help="Video file to test")
    parser.add_argument(
        "--old-script",
        default="video_orientation_detector_old.py",
        help="Path to old version script",
    )
    parser.add_argument(
        "--new-script", default="video_orientation_detector.py", help="Path to new version script"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return 1

    if not os.path.exists(args.old_script):
        print(f"❌ Old script not found: {args.old_script}")
        return 1

    if not os.path.exists(args.new_script):
        print(f"❌ New script not found: {args.new_script}")
        return 1

    print("🎬 Video Orientation Detector Test")
    print(f"Video: {os.path.basename(args.video)}")
    print("=" * 60)

    # Run old version
    old_result = run_single_test(args.old_script, args.video, "OLD VERSION")

    # Run new version
    new_result = run_single_test(args.new_script, args.video, "NEW VERSION")

    # Compare results
    compare_outputs(old_result, new_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
