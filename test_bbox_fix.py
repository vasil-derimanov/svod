#!/usr/bin/env python3
"""
Quick YOLOv10 test after bbox fix
Test a few videos to verify fix is working
"""

import os
import sys
import time

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_orientation_detector import OrientationDetector


def test_fixed_yolov10():
    """Test YOLOv10 after bbox fix"""
    print("🔧 Testing YOLOv10 after bbox fix...")
    print("=" * 50)
    
    # Test videos
    test_videos = [
        r"C:\Users\boris\Bad_Examples\20150911_221520.mp4",
        r"C:\Users\boris\Bad_Examples\P2170127.mp4", 
        r"C:\Users\boris\Bad_Examples\P2270220.mp4"
    ]
    
    results = {}
    
    for i, video_path in enumerate(test_videos, 1):
        filename = os.path.basename(video_path)
        print(f"\n📹 [{i}/{len(test_videos)}] Testing: {filename}")
        
        if not os.path.exists(video_path):
            print(f"   ❌ File not found: {video_path}")
            continue
            
        try:
            # Create fresh detector
            detector = OrientationDetector(confidence_threshold=0.5, time_limit=15.0)
            
            start_time = time.time()
            video_results = detector.process_video(video_path, display=False)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Extract results
            orientation = video_results.get('orientation', 'UNCERTAIN')
            confidence = video_results.get('confidence', 0.0)
            rotation_angle = video_results.get('rotation_angle', 'N/A')
            
            results[filename] = {
                'orientation': orientation,
                'confidence': confidence,
                'rotation_angle': rotation_angle,
                'processing_time': processing_time
            }
            
            print(f"   ✅ Result: {orientation} (confidence: {confidence:.3f})")
            print(f"   🔄 Rotation: {rotation_angle}")
            print(f"   ⏱️ Time: {processing_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[filename] = {'error': str(e)}
    
    # Summary
    print(f"\n📊 Summary:")
    print("=" * 50)
    
    successful_tests = 0
    uncertain_results = 0
    
    for filename, result in results.items():
        if 'error' in result:
            print(f"❌ {filename}: ERROR")
        else:
            orientation = result['orientation']
            confidence = result['confidence']
            time_taken = result['processing_time']
            
            if orientation == 'UNCERTAIN':
                uncertain_results += 1
                print(f"⚠️  {filename}: {orientation} ({confidence:.3f}) - {time_taken:.1f}s")
            else:
                successful_tests += 1
                print(f"✅ {filename}: {orientation} ({confidence:.3f}) - {time_taken:.1f}s")
    
    print(f"\nResults: {successful_tests} successful, {uncertain_results} uncertain")
    
    if uncertain_results == 0:
        print("🎉 bbox fix is working perfectly!")
        return True
    else:
        print("⚠️ Still some issues remain")
        return False


if __name__ == "__main__":
    test_fixed_yolov10()