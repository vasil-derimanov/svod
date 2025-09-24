#!/usr/bin/env python3
"""
Test script to verify YOLOv10 upgrade functionality
Tests the model loading and initialization of YOLOv10 vs YOLOv8
"""

import sys
import os

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_yolov10_import():
    """Test YOLOv10 model import and initialization"""
    print("🧪 Testing YOLOv10 Upgrade...")
    
    try:
        from video_orientation_detector import OrientationDetector
        
        print("✅ Successfully imported OrientationDetector")
        
        # Initialize detector to trigger YOLOv10 model download
        print("🔄 Initializing OrientationDetector (this will download YOLOv10 model if needed)...")
        detector = OrientationDetector(confidence_threshold=0.5, time_limit=10.0)
        
        print("✅ OrientationDetector initialized successfully!")
        
        # Check if YOLOv10 model was loaded
        if hasattr(detector, 'use_yolov10') and detector.use_yolov10:
            print("✅ YOLOv10 model loaded successfully!")
            if hasattr(detector, 'yolov10_model'):
                print("✅ YOLOv10 model attribute exists!")
                model_info = str(detector.yolov10_model)
                print(f"📋 Model info: {model_info[:100]}..." if len(model_info) > 100 else f"📋 Model info: {model_info}")
                return True
            else:
                print("❌ YOLOv10 model attribute missing!")
                return False
        else:
            print("❌ YOLOv10 model not loaded!")
            return False
            
    except Exception as e:
        print(f"❌ Error during YOLOv10 test: {e}")
        import traceback
        print("🔍 Full traceback:")
        traceback.print_exc()
        return False

def test_model_files():
    """Check what model files exist"""
    print("\n📁 Checking model files...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    yolov8_path = os.path.join(script_dir, "yolov8n.pt")
    yolov10_path = os.path.join(script_dir, "yolov10n.pt")
    
    print(f"📋 YOLOv8 model (yolov8n.pt): {'✅ EXISTS' if os.path.exists(yolov8_path) else '❌ MISSING'}")
    print(f"📋 YOLOv10 model (yolov10n.pt): {'✅ EXISTS' if os.path.exists(yolov10_path) else '❌ MISSING'}")
    
    if os.path.exists(yolov8_path):
        size_mb = os.path.getsize(yolov8_path) / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        
    if os.path.exists(yolov10_path):
        size_mb = os.path.getsize(yolov10_path) / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")

if __name__ == "__main__":
    print("🚀 YOLOv10 Upgrade Test Suite")
    print("=" * 50)
    
    # Test model files first
    test_model_files()
    
    # Test YOLOv10 initialization
    success = test_yolov10_import()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 YOLOv10 upgrade test PASSED!")
        
        # Re-check model files after initialization
        print("\n📁 Re-checking model files after initialization...")
        test_model_files()
    else:
        print("💥 YOLOv10 upgrade test FAILED!")
        
    print("=" * 50)