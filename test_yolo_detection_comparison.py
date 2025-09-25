#!/usr/bin/env python3
"""
Direct comparison between YOLOv8 and YOLOv10 person detection
Test the same video with both models to identify the issue
"""

import cv2
import numpy as np
import time
import os
import sys
from ultralytics import YOLO

# Test video path (first file from Bad_Examples)
test_video = r"C:\Users\boris\Bad_Examples\20150911_221520.mp4"

def test_yolo_detection(model_path: str, model_name: str, video_path: str, max_frames: int = 50):
    """Test YOLO model detection on specific video"""
    print(f"\n🧪 Testing {model_name} detection on: {os.path.basename(video_path)}")
    print("=" * 60)
    
    try:
        # Load model
        print(f"📥 Loading {model_name} model: {model_path}")
        model = YOLO(model_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        print(f"📹 Video info: {width}x{height}, {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        # Test detection on frames
        frame_count = 0
        total_detections = 0
        total_confidence = 0
        detection_times = []
        
        frames_to_test = min(max_frames, total_frames)
        frame_interval = max(1, total_frames // frames_to_test)
        
        print(f"🔍 Testing {frames_to_test} frames (every {frame_interval} frames)...")
        
        for i in range(0, total_frames, frame_interval):
            if frame_count >= frames_to_test:
                break
                
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_count += 1
            
            # Run detection
            start_time = time.time()
            results = model(frame, verbose=False)
            end_time = time.time()
            
            detection_time = end_time - start_time
            detection_times.append(detection_time)
            
            # Count person detections (class 0 in COCO)
            frame_detections = 0
            frame_confidence = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # Person class
                            confidence = float(box.conf[0])
                            if confidence > 0.3:  # Minimum confidence threshold
                                frame_detections += 1
                                frame_confidence += confidence
                                
                                # Get bounding box
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                area = (x2 - x1) * (y2 - y1)
                                
                                if frame_count <= 5:  # Show details for first 5 frames
                                    print(f"   Frame {frame_count}: Person detected - conf: {confidence:.3f}, area: {area:.0f}px²")
            
            total_detections += frame_detections
            total_confidence += frame_confidence
            
            if frame_count % 10 == 0:
                print(f"   Processed {frame_count}/{frames_to_test} frames...")
        
        cap.release()
        
        # Calculate statistics
        avg_detection_time = np.mean(detection_times) if detection_times else 0
        avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0
        avg_confidence = total_confidence / max(total_detections, 1)
        detection_rate = (sum(1 for d in detection_times if d > 0) / len(detection_times) * 100) if detection_times else 0
        
        print(f"\n📊 {model_name} Detection Results:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Avg detections/frame: {avg_detections_per_frame:.2f}")
        print(f"   Avg confidence: {avg_confidence:.3f}")
        print(f"   Avg detection time: {avg_detection_time:.3f}s")
        print(f"   Detection rate: {detection_rate:.1f}% of frames")
        
        return {
            'model_name': model_name,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'avg_confidence': avg_confidence,
            'avg_detection_time': avg_detection_time,
            'detection_rate': detection_rate
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main comparison test"""
    print("🚀 YOLOv8 vs YOLOv10 Person Detection Comparison")
    print("Testing on Bad_Examples video to identify detection issues")
    print("=" * 70)
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return
    
    # Test YOLOv8
    yolov8_results = test_yolo_detection("yolov8n.pt", "YOLOv8", test_video)
    
    # Test YOLOv10  
    yolov10_results = test_yolo_detection("yolov10n.pt", "YOLOv10", test_video)
    
    # Compare results
    if yolov8_results and yolov10_results:
        print(f"\n🏆 Detection Comparison Summary:")
        print("=" * 60)
        
        print(f"{'Metric':<25} {'YOLOv8':<15} {'YOLOv10':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Detections per frame
        v8_dpf = yolov8_results['avg_detections_per_frame']
        v10_dpf = yolov10_results['avg_detections_per_frame']
        dpf_diff = v10_dpf - v8_dpf
        dpf_pct = ((v10_dpf - v8_dpf) / max(v8_dpf, 0.001)) * 100
        
        print(f"{'Detections/Frame':<25} {v8_dpf:<15.2f} {v10_dpf:<15.2f} {dpf_diff:+.2f} ({dpf_pct:+.1f}%)")
        
        # Total detections
        v8_total = yolov8_results['total_detections']
        v10_total = yolov10_results['total_detections']
        total_diff = v10_total - v8_total
        
        print(f"{'Total Detections':<25} {v8_total:<15} {v10_total:<15} {total_diff:+d}")
        
        # Confidence
        v8_conf = yolov8_results['avg_confidence']
        v10_conf = yolov10_results['avg_confidence']
        conf_diff = v10_conf - v8_conf
        
        print(f"{'Avg Confidence':<25} {v8_conf:<15.3f} {v10_conf:<15.3f} {conf_diff:+.3f}")
        
        # Detection rate
        v8_rate = yolov8_results['detection_rate']
        v10_rate = yolov10_results['detection_rate']
        rate_diff = v10_rate - v8_rate
        
        print(f"{'Detection Rate %':<25} {v8_rate:<15.1f} {v10_rate:<15.1f} {rate_diff:+.1f}")
        
        print("-" * 70)
        
        # Conclusion
        if v10_dpf < v8_dpf * 0.5:  # YOLOv10 detects less than half
            print("🔴 ISSUE IDENTIFIED: YOLOv10 detects significantly fewer persons than YOLOv8!")
            print("   This explains why SVOD falls back to face-only detection and returns UNCERTAIN.")
        elif v10_dpf > v8_dpf * 1.2:  # YOLOv10 detects 20% more
            print("🟢 YOLOv10 detects more persons than YOLOv8 - this should improve accuracy!")
        else:
            print("🟡 YOLOv10 and YOLOv8 have similar person detection rates.")
        
        return yolov8_results, yolov10_results
    else:
        print("💥 Comparison failed - could not test both models!")
        return None, None


if __name__ == "__main__":
    main()