#!/usr/bin/env python3
"""
Test YOLOv10 vs YOLOv8 performance comparison
Creates a simple test to compare detection capabilities
"""

import cv2
import numpy as np
import time
import os
import tempfile


def create_test_frame_with_person():
    """Create a test frame with a simple person-like rectangle"""
    # Create a 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some background noise
    frame[:, :] = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    
    # Draw a simple "person" - rectangle with head
    person_color = (100, 150, 200)  # BGR
    
    # Body (rectangle)
    cv2.rectangle(frame, (250, 200), (390, 450), person_color, -1)
    
    # Head (circle)
    cv2.circle(frame, (320, 150), 40, person_color, -1)
    
    # Add some detail to make it more recognizable
    # Arms
    cv2.rectangle(frame, (210, 220), (250, 320), person_color, -1)  # Left arm
    cv2.rectangle(frame, (390, 220), (430, 320), person_color, -1)  # Right arm
    
    # Legs
    cv2.rectangle(frame, (270, 450), (310, 470), person_color, -1)  # Left leg
    cv2.rectangle(frame, (330, 450), (370, 470), person_color, -1)  # Right leg
    
    return frame


def create_test_video():
    """Create a simple test video with a person"""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "test_person_video.mp4")
    
    # Video properties
    fps = 10
    duration_seconds = 3
    frame_count = fps * duration_seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
    
    print(f"📹 Creating test video: {video_path}")
    print(f"   Duration: {duration_seconds}s, FPS: {fps}, Frames: {frame_count}")
    
    for i in range(frame_count):
        frame = create_test_frame_with_person()
        
        # Add frame number
        cv2.putText(frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    
    # Verify file was created
    if os.path.exists(video_path):
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"✅ Test video created successfully!")
        print(f"   Size: {file_size_mb:.1f} MB")
        return video_path
    else:
        print("❌ Failed to create test video!")
        return None


def test_yolov10_detection(video_path):
    """Test YOLOv10 detection on our test video"""
    print(f"\n🧪 Testing YOLOv10 detection on: {video_path}")
    
    try:
        # Import our detector
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from video_orientation_detector import OrientationDetector
        
        # Create detector with short time limit
        detector = OrientationDetector(confidence_threshold=0.3, time_limit=2.0)
        
        print(f"🔍 Analyzing video with YOLOv10 (time limit: 2.0s)...")
        start_time = time.time()
        
        results = detector.process_video(video_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️ Processing completed in {processing_time:.2f}s")
        
        # Display results
        if results:
            print("📊 Detection Results:")
            print(f"   Confidence: {results.get('confidence', 'N/A')}")
            print(f"   Rotation: {results.get('rotation_angle', 'N/A')}°")
            print(f"   Face detections: {results.get('face_count', 0)}")
            print(f"   Person detections: {results.get('person_count', 0)}")
            print(f"   Processing method: {results.get('method', 'N/A')}")
            
            # Check detection statistics
            if 'stats' in results:
                stats = results['stats']
                print("📈 Detection Statistics:")
                for key, value in stats.items():
                    if 'count' in key.lower():
                        print(f"   {key}: {value}")
            
            return True
        else:
            print("❌ No results returned!")
            return False
            
    except Exception as e:
        print(f"❌ Error during YOLOv10 detection test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 YOLOv10 Detection Test Suite")
    print("=" * 50)
    
    # Create test video
    video_path = create_test_video()
    
    if not video_path:
        print("💥 Cannot proceed without test video!")
        return False
    
    # Test YOLOv10 detection
    success = test_yolov10_detection(video_path)
    
    # Cleanup
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"🗑️ Cleaned up test video: {video_path}")
    except Exception as e:
        print(f"⚠️ Could not clean up test video: {e}")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 YOLOv10 detection test PASSED!")
    else:
        print("💥 YOLOv10 detection test FAILED!")
    
    return success


if __name__ == "__main__":
    main()