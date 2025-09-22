#!/usr/bin/env python3
"""
Test Bad_Examples videos one by one and extract orientation suggestions
"""
import os
import subprocess
import sys
import re

def test_video_simple(video_path):
    """Test video and extract results using simple text parsing"""
    try:
        # Set environment to avoid unicode issues
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, 'video_orientation_detector.py', video_path, '--time-limit', '8'],
            capture_output=True,
            text=True,
            env=env,
            errors='replace',
            timeout=180
        )
        
        output = result.stdout + result.stderr
        
        # Extract orientation and recommendation using regex
        orientation = "UNKNOWN"
        recommendation = "No recommendation found"
        confidence = "0%"
        
        # Look for orientation status
        if re.search(r'\[ERROR\]\s*INCORRECT', output):
            orientation = "INCORRECT"
        elif re.search(r'\[OK\]\s*CORRECT', output):
            orientation = "CORRECT"
        elif 'UNCERTAIN' in output:
            orientation = "UNCERTAIN"
            
        # Look for recommendation
        recommendation_match = re.search(r'Recommendation:\s*([^\n\r]+)', output)
        if recommendation_match:
            recommendation = recommendation_match.group(1).strip()
        
        # Look for confidence
        confidence_match = re.search(r'Confidence:\s*([0-9.]+%)', output)
        if confidence_match:
            confidence = confidence_match.group(1)
            
        return orientation, recommendation, confidence, True
        
    except Exception as e:
        return "ERROR", f"Failed: {str(e)}", "0%", False

def main():
    bad_examples_dir = r"C:\Users\boris\Bad_Examples"
    
    print("=" * 100)
    print("BAD_EXAMPLES DIRECTORY - DETAILED ORIENTATION SUGGESTIONS")
    print("=" * 100)
    
    if not os.path.exists(bad_examples_dir):
        print(f"Directory not found: {bad_examples_dir}")
        return
    
    # Get all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    video_files = []
    
    for file in os.listdir(bad_examples_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    video_files.sort()
    
    print(f"\nFound {len(video_files)} video files to test:")
    print("-" * 100)
    
    # Reference expectations
    reference = {
        "P2170127.mp4": "needs 90° clockwise rotation",
        "P6160117.mp4": "needs 90° clockwise rotation", 
        "P9080828.mp4": "needs 90° counterclockwise rotation",
        "VID_20200907_202511.mp4": "needs 90° counterclockwise rotation",
        "20150911_221520.mp4": "needs 90° counterclockwise rotation",
        "P7210301.mp4": "needs 90° counterclockwise rotation"
    }
    
    successful_tests = 0
    total_tests = len(video_files)
    
    for i, filename in enumerate(video_files, 1):
        video_path = os.path.join(bad_examples_dir, filename)
        
        print(f"\n📹 Video {i:2d}/{total_tests}: {filename}")
        
        # Show reference expectation if available
        if filename in reference:
            print(f"📋 Reference: {reference[filename]}")
        
        orientation, recommendation, confidence, success = test_video_simple(video_path)
        
        if success:
            successful_tests += 1
            print(f"✅ Status: {orientation}")
            print(f"🎯 Recommendation: {recommendation}")
            print(f"📊 Confidence: {confidence}")
            
            # Check if it matches reference expectation
            if filename in reference:
                ref_expectation = reference[filename]
                if "counterclockwise" in ref_expectation and "counterclockwise" in recommendation:
                    print("✅ MATCHES REFERENCE")
                elif "clockwise" in ref_expectation and "clockwise" in recommendation and "counterclockwise" not in recommendation:
                    print("✅ MATCHES REFERENCE")
                else:
                    print("⚠️  CHECK: May not match reference expectation")
        else:
            print(f"❌ Status: {orientation}")
            print(f"⚠️  Issue: {recommendation}")
        
        print("-" * 60)
    
    print(f"\n📊 SUMMARY:")
    print(f"Total videos tested: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    print("=" * 100)

if __name__ == "__main__":
    main()