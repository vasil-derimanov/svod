#!/usr/bin/env python3
"""
Performance comparison between YOLOv8 and YOLOv10
Tests detection speed and accuracy between the two models
"""

import cv2
import numpy as np
import time
import os
import tempfile
import sys

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO


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


def test_yolo_model_performance(model_name: str, model_file: str, test_frames: int = 10):
    """Test YOLO model performance on synthetic frames"""
    print(f"\n🧪 Testing {model_name} performance...")
    
    try:
        # Load model
        print(f"📥 Loading {model_name} model: {model_file}")
        model = YOLO(model_file)
        print(f"✅ {model_name} model loaded successfully!")
        
        # Generate test frames
        print(f"🎬 Generating {test_frames} test frames...")
        frames = []
        for i in range(test_frames):
            frame = create_test_frame_with_person()
            # Vary the person position slightly
            offset = i * 10
            frame = np.roll(frame, offset, axis=1)
            frames.append(frame)
        
        # Performance test
        print(f"⏱️ Running {model_name} detection on {test_frames} frames...")
        start_time = time.time()
        
        total_detections = 0
        total_confidence = 0
        detection_times = []
        
        for i, frame in enumerate(frames):
            frame_start = time.time()
            
            # Run detection
            results = model(frame, verbose=False)
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            detection_times.append(frame_time)
            
            # Count person detections (class 0 in COCO)
            frame_detections = 0
            frame_confidence = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # Person class
                            confidence = float(box.conf[0])
                            if confidence > 0.3:  # Same threshold as our detector
                                frame_detections += 1
                                frame_confidence += confidence
            
            total_detections += frame_detections
            total_confidence += frame_confidence
            
            if (i + 1) % 5 == 0:
                print(f"   Processed {i+1}/{test_frames} frames...")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        avg_detection_time = np.mean(detection_times)
        fps = 1.0 / avg_detection_time
        avg_confidence = total_confidence / max(total_detections, 1)
        
        results = {
            'model_name': model_name,
            'total_time': total_time,
            'avg_detection_time': avg_detection_time,
            'fps': fps,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / test_frames,
            'avg_confidence': avg_confidence,
            'detection_times': detection_times
        }
        
        # Print results
        print(f"📊 {model_name} Performance Results:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Avg detection time: {avg_detection_time:.3f}s per frame")
        print(f"   FPS: {fps:.1f}")
        print(f"   Total detections: {total_detections}")
        print(f"   Avg detections per frame: {results['avg_detections_per_frame']:.1f}")
        print(f"   Avg confidence: {avg_confidence:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_performance():
    """Compare YOLOv8 vs YOLOv10 performance"""
    print("🚀 YOLOv8 vs YOLOv10 Performance Comparison")
    print("=" * 60)
    
    test_frames = 20
    
    # Test YOLOv8
    yolov8_results = test_yolo_model_performance("YOLOv8", "yolov8n.pt", test_frames)
    
    # Test YOLOv10
    yolov10_results = test_yolo_model_performance("YOLOv10", "yolov10n.pt", test_frames)
    
    # Compare results
    if yolov8_results and yolov10_results:
        print(f"\n🏆 Performance Comparison Summary:")
        print("=" * 60)
        
        print(f"{'Metric':<25} {'YOLOv8':<15} {'YOLOv10':<15} {'Winner':<10}")
        print("-" * 60)
        
        # Speed comparison
        v8_fps = yolov8_results['fps']
        v10_fps = yolov10_results['fps']
        speed_winner = "YOLOv10" if v10_fps > v8_fps else "YOLOv8" if v8_fps > v10_fps else "Tie"
        print(f"{'FPS (Speed)':<25} {v8_fps:<15.1f} {v10_fps:<15.1f} {speed_winner:<10}")
        
        # Detection time comparison
        v8_time = yolov8_results['avg_detection_time']
        v10_time = yolov10_results['avg_detection_time']
        time_winner = "YOLOv10" if v10_time < v8_time else "YOLOv8" if v8_time < v10_time else "Tie"
        print(f"{'Avg Detection Time (s)':<25} {v8_time:<15.3f} {v10_time:<15.3f} {time_winner:<10}")
        
        # Detection accuracy comparison
        v8_detections = yolov8_results['avg_detections_per_frame']
        v10_detections = yolov10_results['avg_detections_per_frame']
        detection_winner = "YOLOv10" if v10_detections > v8_detections else "YOLOv8" if v8_detections > v10_detections else "Tie"
        print(f"{'Detections/Frame':<25} {v8_detections:<15.1f} {v10_detections:<15.1f} {detection_winner:<10}")
        
        # Confidence comparison
        v8_conf = yolov8_results['avg_confidence']
        v10_conf = yolov10_results['avg_confidence']
        conf_winner = "YOLOv10" if v10_conf > v8_conf else "YOLOv8" if v8_conf > v10_conf else "Tie"
        print(f"{'Avg Confidence':<25} {v8_conf:<15.3f} {v10_conf:<15.3f} {conf_winner:<10}")
        
        print("-" * 60)
        
        # Overall winner
        v10_wins = sum([
            v10_fps > v8_fps,  # Speed
            v10_time < v8_time,  # Detection time (lower is better)
            v10_detections > v8_detections,  # More detections
            v10_conf > v8_conf,  # Higher confidence
        ])
        
        v8_wins = 4 - v10_wins
        
        if v10_wins > v8_wins:
            overall_winner = "YOLOv10"
            print(f"🎉 Overall Winner: {overall_winner} ({v10_wins}/4 metrics)")
        elif v8_wins > v10_wins:
            overall_winner = "YOLOv8"
            print(f"🎉 Overall Winner: {overall_winner} ({v8_wins}/4 metrics)")
        else:
            print(f"🤝 Overall Result: Tie ({v8_wins}-{v10_wins})")
        
        # Performance improvement
        if v10_wins > v8_wins:
            speed_improvement = ((v10_fps - v8_fps) / v8_fps) * 100
            time_improvement = ((v8_time - v10_time) / v8_time) * 100
            print(f"📈 YOLOv10 Improvements:")
            print(f"   Speed: {speed_improvement:+.1f}% FPS")
            print(f"   Detection Time: {time_improvement:+.1f}% faster")
        
        return yolov8_results, yolov10_results
    else:
        print("💥 Performance comparison failed!")
        return None, None


if __name__ == "__main__":
    compare_performance()