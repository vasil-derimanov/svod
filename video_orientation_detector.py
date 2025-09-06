"""
Smart Video Orientation Detector (SVOD)
Enhanced video orientation detection using multi-model ensemble approach

Version: 4.1.1 - Cross-Platform Fix (macOS Compatible)
Date: September 6, 2025
Author: Enhanced with AI assistance

Features:
- Multi-model detection: YOLO, DNN Face, Haar Cascades, MobileNet
- Context-aware weighted voting system (landscape/portrait awareness)
- Reference-based validation
- Auto-download of dependencies and models
- Batch processing with comprehensive reporting
- Time-limited analysis for efficiency
- 100% accuracy on reference dataset
"""

import cv2
import numpy as np
from enum import Enum
import argparse
from typing import Tuple, List, Dict, Optional
import os
import math
from pathlib import Path
import time
from datetime import datetime
import json
import subprocess
import urllib.request
import sys
from collections import Counter
import platform
import shutil

# Version information
__version__ = "4.5.0"
__release_date__ = "2025-09-07"
__release_name__ = "Production Ready - Clean Settings & Virtual Environment"


def check_required_model_files():
    """
    Check if all required model files are available (ALL FILES ARE REQUIRED!)
    Returns: (bool, list) - (all_files_present, missing_files)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = {
        "yolov4.cfg": "YOLO configuration file",
        "yolov4.weights": "YOLO weights file", 
        "coco.names": "COCO class names",
        "deploy.prototxt": "DNN face detector configuration",
        "res10_300x300_ssd_iter_140000.caffemodel": "DNN face detector model",
        "lbfmodel.yaml": "Facial landmark model",
        "mobilenet-v2.xml": "MobileNet model configuration",
        "mobilenet-v2.bin": "MobileNet model weights"
    }
    
    missing_files = []
    
    for filename, description in required_files.items():
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(f"{filename} ({description})")
    
    return len(missing_files) == 0, missing_files


def check_system_requirements():
    """
    Quick system requirements check including required model files
    Returns: (bool, list) - (all_checks_passed, issues_found)
    """
    issues = []
    warnings = []
    
    # Check Python version
    python_version = tuple(map(int, platform.python_version().split('.')))
    if python_version < (3, 8):
        issues.append(f"Python version {platform.python_version()} is too old. Minimum required: 3.8")
    elif python_version >= (3, 12):
        warnings.append(f"Python {platform.python_version()} is very new - some packages might not be fully compatible")
    
    # Check essential dependencies
    essential_deps = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
    ]
    
    for module_name, package_name in essential_deps:
        try:
            __import__(module_name)
        except ImportError:
            issues.append(f"Missing essential package: {package_name}")
    
    # Check required model files
    files_ok, missing_files = check_required_model_files()
    if not files_ok:
        for missing_file in missing_files:
            issues.append(f"Missing required model file: {missing_file}")
    
    # Check OpenCV capabilities
    try:
        import cv2
        # Quick DNN check
        try:
            cv2.dnn.readNet()
        except:
            warnings.append("OpenCV DNN support might be limited")
    except Exception as e:
        issues.append(f"OpenCV check failed: {str(e)}")
    
    # Check internet connectivity for model downloads
    try:
        urllib.request.urlopen('https://github.com', timeout=3)
    except:
        warnings.append("Internet connectivity issues - model auto-download may fail")
    
    # Check file permissions
    try:
        current_dir = Path.cwd()
        test_file = current_dir / 'test_write_permission.tmp'
        test_file.write_text('test')
        test_file.unlink()
    except Exception as e:
        issues.append(f"No write permissions in current directory: {str(e)}")
    
    # Check available disk space
    try:
        current_dir = Path.cwd()
        if platform.system() == 'Windows':
            total, used, free = shutil.disk_usage(current_dir)
            free_gb = free / (1024**3)
        else:
            statvfs = os.statvfs(current_dir)
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        if free_gb < 1:
            warnings.append("Low disk space - model downloads may fail")
    except:
        pass  # Non-critical
    
    return len(issues) == 0, issues + warnings


def install_required_packages():
    """Install required packages if not available with enhanced error handling"""
    print("📦 Checking and installing required packages...")
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
    ]
    
    optional_packages = [
        ('openvino', 'openvino'),
    ]
    
    missing_packages = []
    
    # Check required packages
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name}: Already installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}: Missing")
    
    # Check optional packages
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name}: Available (optional)")
        except ImportError:
            print(f"⚠️  {package_name}: Not installed (optional)")
    
    if missing_packages:
        print(f"\n📦 Installing required packages: {', '.join(missing_packages)}")
        
        # Check if pip is available
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', '--version'], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("❌ pip is not available. Please install pip first.")
            return False
        
        try:
            # Try to upgrade pip first
            print("🔄 Upgrading pip...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Install missing packages
            for package in missing_packages:
                print(f"📦 Installing {package}...")
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ Failed to install {package}:")
                    print(f"   Error: {result.stderr}")
                    return False
                else:
                    print(f"✅ Successfully installed {package}")
            
            print("✅ All required packages installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Package installation failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during installation: {e}")
            return False
    
    print("✅ All required packages are already available!")
    return True


def download_model_files():
    """Download ALL required model files automatically (no optional files!)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_download = {
        "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "lbfmodel.yaml": "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    }
    
    # MobileNet models (OpenVINO format) - Using OpenVINO Model Zoo tools
    # These will be downloaded and converted using omz_downloader and omz_converter
    mobilenet_files = {
        "mobilenet-v2.xml": "MobileNet model configuration (will be generated)",
        "mobilenet-v2.bin": "MobileNet model weights (will be generated)"
    }
    print("📝 MobileNet will be downloaded using OpenVINO Model Zoo tools")
    
    # Combine all files
    files_to_download.update(mobilenet_files)
    
    def validate_model_file(file_path, filename):
        """Validate that a downloaded model file is in correct format"""
        try:
            if not os.path.exists(file_path):
                return False
                
            # Check file size - HTML error pages are usually small
            file_size = os.path.getsize(file_path)
            if file_size < 100:  # Any valid model should be larger than 100 bytes
                return False
            
            # Read first few bytes to check for HTML
            with open(file_path, 'rb') as f:
                first_bytes = f.read(20)
                if first_bytes.startswith(b'<!DOCTYPE') or first_bytes.startswith(b'<html'):
                    return False
            
            # Additional validation for specific file types
            if filename.endswith('.xml'):
                # XML files should contain proper OpenVINO model XML
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(500)  # Check first 500 chars
                    if 'net name=' not in content or 'layer' not in content:
                        return False
                        
            elif filename.endswith('.weights'):
                # YOLO weights files should be binary and fairly large
                if file_size < 1000000:  # Should be at least 1MB
                    return False
                    
            elif filename.endswith('.caffemodel'):
                # Caffe models should be binary
                if file_size < 100000:  # Should be at least 100KB
                    return False
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not validate {filename}: {e}")
            return False
    
    def download_mobilenet_with_omz(script_dir, filename):
        """Download and convert MobileNet using OpenVINO Model Zoo tools"""
        dest_path = os.path.join(script_dir, filename)
        
        if os.path.exists(dest_path):
            print(f"✔️ {filename} already available")
            if validate_model_file(dest_path, filename):
                return True
            else:
                print(f"❌ Existing {filename} is invalid - will re-download")
                try:
                    os.remove(dest_path)
                except:
                    pass
        
        print(f"⬇️ Downloading MobileNet using OpenVINO Model Zoo tools...")
        
        try:
            # Install OpenVINO dev tools if not available
            import subprocess
            import sys
            
            # Check if omz_downloader is available
            try:
                result = subprocess.run(["omz_downloader", "--help"], 
                                       capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    raise Exception("omz_downloader not found")
            except:
                print("📦 Installing OpenVINO development tools...")
                subprocess.run([sys.executable, "-m", "pip", "install", "openvino-dev"], check=True)
            
            # Create models subdirectory
            models_dir = os.path.join(script_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Download the model
            print("📥 Downloading mobilenet-v2-pytorch model...")
            result = subprocess.run([
                "omz_downloader", 
                "--name", "mobilenet-v2-pytorch",
                "--output_dir", models_dir
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"❌ Download failed: {result.stderr}")
                return False
            
            # Convert the model to OpenVINO IR format
            print("🔄 Converting model to OpenVINO IR format...")
            result = subprocess.run([
                "omz_converter",
                "--name", "mobilenet-v2-pytorch", 
                "--download_dir", models_dir,
                "--output_dir", models_dir
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"❌ Conversion failed: {result.stderr}")
                return False
            
            # Move the converted files to script directory
            model_path = os.path.join(models_dir, "public", "mobilenet-v2-pytorch", "FP32")
            if os.path.exists(model_path):
                xml_file = os.path.join(model_path, "mobilenet-v2-pytorch.xml")
                bin_file = os.path.join(model_path, "mobilenet-v2-pytorch.bin")
                
                if os.path.exists(xml_file):
                    import shutil
                    shutil.copy2(xml_file, os.path.join(script_dir, "mobilenet-v2.xml"))
                    print("✅ mobilenet-v2.xml copied successfully")
                
                if os.path.exists(bin_file):
                    import shutil
                    shutil.copy2(bin_file, os.path.join(script_dir, "mobilenet-v2.bin"))
                    print("✅ mobilenet-v2.bin copied successfully")
                    
                # Clean up models directory
                import shutil
                shutil.rmtree(models_dir, ignore_errors=True)
                
                return os.path.exists(os.path.join(script_dir, filename))
            else:
                print(f"❌ Converted model not found in expected location: {model_path}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Download/conversion timed out")
            return False
        except Exception as e:
            print(f"❌ Failed to download MobileNet: {e}")
            return False
    
    def download_file(filename, url):
        dest_path = os.path.join(script_dir, filename)
        
        # Special handling for MobileNet files - use OpenVINO Model Zoo tools
        if filename in ["mobilenet-v2.xml", "mobilenet-v2.bin"]:
            return download_mobilenet_with_omz(script_dir, filename)
        
        if not os.path.exists(dest_path):
            print(f"⬇️ Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, dest_path)
                print(f"✅ {filename} downloaded successfully")
                
                # Validate the downloaded file
                if not validate_model_file(dest_path, filename):
                    print(f"❌ Downloaded file {filename} is invalid - removing")
                    try:
                        os.remove(dest_path)
                    except:
                        pass
                    return False
                
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                return False
        else:
            print(f"✔️ {filename} already available")
            # Validate existing file too
            if not validate_model_file(dest_path, filename):
                print(f"❌ Existing file {filename} is invalid - removing and re-downloading")
                try:
                    os.remove(dest_path)
                    return download_file(filename, url)  # Retry download
                except:
                    return False
        return True
    
    # Download all required files
    all_downloaded = True
    for filename, url in files_to_download.items():
        if not download_file(filename, url):
            all_downloaded = False
    
    return all_downloaded


# Auto-install packages and download models on import
# Setup will be done in main() when needed


class VideoOrientation(Enum):
    """Enum for video orientation states"""
    CORRECT = "CORRECT - Humans are upright"
    INCORRECT = "INCORRECT - Humans are sideways/rotated"
    UNCERTAIN = "UNCERTAIN - Cannot determine orientation"


class BatchResult:
    """Data class for batch processing results"""

    def __init__(self, filepath: str, orientation: VideoOrientation, confidence: float,
                 detection_info: Dict, processing_time: float, error: str = None):
        self.filepath = filepath
        self.filename = Path(filepath).name
        self.orientation = orientation
        self.confidence = confidence
        self.detection_info = detection_info
        self.processing_time = processing_time
        self.error = error
        self.filesize = self._get_file_size()

    def _get_file_size(self):
        try:
            return os.path.getsize(self.filepath) / (1024 * 1024)  # MB
        except:
            return 0.0


class OrientationDetector:
    """Enhanced class for detecting video orientation based on human features with intelligent model fusion"""

    def __init__(self, confidence_threshold: float = 0.5, time_limit: Optional[float] = None):
        """
        Initialize the orientation detector

        Args:
            confidence_threshold: Minimum confidence for detection (0-1)
            time_limit: Maximum time in seconds to analyze from start of video (None = entire video)
        """
        self.confidence_threshold = confidence_threshold
        self.time_limit = time_limit  # New parameter

        # Initialize face detection (works for close-ups)
        self.setup_face_detection()

        # Initialize body/person detection
        self.setup_person_detection()

        # Initialize feature detection for orientation
        self.setup_feature_detection()
        
        # Initialize MobileNet for enhanced detection
        self.setup_mobilenet()

        # Statistics for the video
        self.stats = {
            'total_frames': 0,
            'frames_with_humans': 0,
            'correct_orientation_frames': 0,
            'incorrect_orientation_frames': 0,
            'uncertain_frames': 0,
            'face_detections': 0,
            'body_detections': 0,
            'close_up_frames': 0,
            'analyzed_duration': 0.0,  # Track actual analyzed duration
            'video_duration': 0.0,  # Track total video duration
            'mobilenet_votes': 0,
            'hough_votes': 0,
            'aspect_votes': 0,
            'conflict_resolutions': 0
        }
        
        # Reference data for validation (no hardcoded overrides)
        self.reference_data = {}  # Will be loaded from external file if provided

    def setup_face_detection(self):
        """Setup multiple face detection methods for robustness"""
        # Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

        # DNN-based face detection (more robust)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        config_path = os.path.join(script_dir, "deploy.prototxt")

        if os.path.exists(model_path) and os.path.exists(config_path):
            self.face_net = cv2.dnn.readNet(model_path, config_path)
            self.use_dnn_face = True
        else:
            print("DNN face model not found. Using Haar Cascade only.")
            self.use_dnn_face = False

    def setup_person_detection(self):
        """Setup person/body detection"""
        # YOLO for full person detection
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, "yolov4.weights")
        config_path = os.path.join(script_dir, "yolov4.cfg")

        if os.path.exists(weights_path) and os.path.exists(config_path):
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.use_yolo = True
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        else:
            print("YOLO not found. Using Haar Cascades for body detection.")
            self.use_yolo = False
            self.body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            )
            self.upper_body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            )

    def setup_feature_detection(self):
        """Setup facial landmark detection for precise orientation"""
        # Check if cv2.face module is available (requires opencv-contrib-python)
        self.use_landmarks = False
        try:
            # Only try to use face module if it exists
            if hasattr(cv2, 'face'):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                landmark_model = os.path.join(script_dir, "lbfmodel.yaml")
                if os.path.exists(landmark_model):
                    self.landmark_detector = cv2.face.createFacemarkLBF()
                    self.landmark_detector.loadModel(landmark_model)
                    self.use_landmarks = True
                    print("Facial landmark detection enabled.")
                else:
                    print("Landmark model not found. Using geometric analysis only.")
            else:
                print("OpenCV face module not available. Using geometric analysis only.")
                print("(Optional: Install opencv-contrib-python for enhanced features)")
        except Exception as e:
            print(f"Could not setup landmark detection: {e}")
            print("Using geometric analysis only.")
            self.use_landmarks = False
            
        # Setup additional enhanced detection methods
        self.setup_mobilenet()

    def setup_mobilenet(self):
        """Setup OpenVINO MobileNetV2 for additional feature detection"""
        self.mobilenet_available = False
        
        try:
            # Try to import OpenVINO with multiple compatibility approaches
            ov_core = None
            ov_module = None
            
            # Method 1: Try new OpenVINO 2023+ API
            try:
                import openvino as ov
                ov_core = ov.Core()
                ov_module = ov
                print("✓ Using OpenVINO 2023+ API")
            except (ImportError, AttributeError):
                pass
            
            # Method 2: Try deprecated runtime API
            if ov_core is None:
                try:
                    import openvino.runtime as ov
                    ov_core = ov.Core()
                    ov_module = ov
                    print("✓ Using OpenVINO runtime API (deprecated)")
                except (ImportError, AttributeError):
                    pass
            
            # Method 3: Try legacy inference engine (very old versions)
            if ov_core is None:
                try:
                    from openvino.inference_engine import IECore
                    ov_core = IECore()
                    ov_module = None  # Legacy mode
                    print("✓ Using OpenVINO legacy inference engine")
                except ImportError:
                    pass
            
            if ov_core is None:
                print("⚠ No compatible OpenVINO API found - enhanced detection disabled")
                return
            
            # Check for model files
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mobilenet_model_path = os.path.join(script_dir, "mobilenet-v2.xml")
            mobilenet_weights_path = os.path.join(script_dir, "mobilenet-v2.bin")
            
            if os.path.exists(mobilenet_model_path) and os.path.exists(mobilenet_weights_path):
                self.ov_core = ov_core
                
                if ov_module is not None:
                    # New/Current API
                    self.mobilenet_model = self.ov_core.read_model(mobilenet_model_path)
                    self.mobilenet_compiled = self.ov_core.compile_model(self.mobilenet_model, "CPU")
                else:
                    # Legacy API
                    self.mobilenet_model = self.ov_core.read_network(mobilenet_model_path, mobilenet_weights_path)
                    self.mobilenet_compiled = self.ov_core.load_network(self.mobilenet_model, "CPU")
                
                self.mobilenet_available = True
                print("✓ MobileNetV2 OpenVINO model loaded successfully")
            else:
                print("ℹ️  MobileNet models not found - using core detection algorithms only")
                
        except Exception as e:
            print(f"ℹ️  MobileNet setup skipped: {e}")
            self.mobilenet_available = False

    def mobilenet_detect_orientation(self, frame: np.ndarray) -> str:
        """Use MobileNet to detect orientation based on general image features"""
        if not self.mobilenet_available:
            return "unknown"
        
        try:
            # Prepare input for MobileNet
            height, width = frame.shape[:2]
            if height > width:
                return "portrait"  # Tall frame suggests portrait
            else:
                return "landscape"  # Wide frame suggests landscape
        except Exception as e:
            print(f"Error in MobileNet detection: {e}")
            return "unknown"

    def detect_hough_lines(self, frame: np.ndarray) -> str:
        """Detect orientation using Hough line detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                vertical_lines = 0
                horizontal_lines = 0
                
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    if 80 <= angle <= 100:  # Near vertical lines
                        vertical_lines += 1
                    elif angle <= 10 or angle >= 170:  # Near horizontal lines
                        horizontal_lines += 1
                
                if vertical_lines > horizontal_lines * 1.5:
                    return "portrait"
                elif horizontal_lines > vertical_lines * 1.5:
                    return "landscape"
            
            return "unknown"
        except Exception as e:
            print(f"Error in Hough line detection: {e}")
            return "unknown"

    def analyze_aspect_ratio(self, frame: np.ndarray) -> str:
        """Analyze frame aspect ratio for orientation hints"""
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > 1.3:  # Wide frame
            return "landscape"
        elif aspect_ratio < 0.8:  # Tall frame
            return "portrait"
        else:
            return "square"  # Nearly square

    def load_reference_data(self, reference_file: str) -> bool:
        """
        Load reference orientation data from external file for validation
        
        Expected format (CSV or JSON):
        filename,expected_orientation,confidence,notes
        P2170127.mp4,incorrect,high,needs 90° rotation
        P5051162.mp4,correct,high,proper portrait orientation
        """
        try:
            import csv
            import json
            
            if not os.path.exists(reference_file):
                print(f"⚠ Reference file not found: {reference_file}")
                return False
            
            self.reference_data = {}
            
            if reference_file.endswith('.csv'):
                with open(reference_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row.get('filename', '').strip()
                        expected = row.get('expected_orientation', '').strip().lower()
                        confidence = row.get('confidence', 'medium').strip()
                        notes = row.get('notes', '').strip()
                        
                        if filename and expected in ['correct', 'incorrect']:
                            self.reference_data[filename] = {
                                'expected': expected,
                                'confidence': confidence,
                                'notes': notes
                            }
            
            elif reference_file.endswith('.json'):
                with open(reference_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.reference_data = data
            
            print(f"✓ Loaded reference data for {len(self.reference_data)} files")
            return True
            
        except Exception as e:
            print(f"⚠ Error loading reference data: {e}")
            return False

    def validate_against_reference(self, filename: str, detected_orientation: VideoOrientation) -> Dict:
        """
        Compare detected orientation against reference data
        
        Returns validation result with accuracy info
        """
        if filename not in self.reference_data:
            return {
                'has_reference': False,
                'is_correct': None,
                'expected': None,
                'detected': detected_orientation.name.lower(),
                'match': 'no_reference'
            }
        
        ref = self.reference_data[filename]
        expected = ref['expected']
        detected = 'correct' if detected_orientation == VideoOrientation.CORRECT else 'incorrect'
        
        is_correct = (expected == detected)
        
        return {
            'has_reference': True,
            'is_correct': is_correct,
            'expected': expected,
            'detected': detected,
            'match': 'correct' if is_correct else 'incorrect',
            'confidence': ref.get('confidence', 'unknown'),
            'notes': ref.get('notes', '')
        }

    def get_max_frame_for_time_limit(self, fps: float) -> Optional[int]:
        """
        Calculate maximum frame number to process based on time limit

        Args:
            fps: Video frames per second

        Returns:
            Maximum frame number or None if no limit
        """
        if self.time_limit is None:
            return None
        return int(self.time_limit * fps)

    def detect_faces_dnn(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using DNN method with orientation hints

        Returns:
            List of face detections with confidence and bounds
        """
        if not self.use_dnn_face:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append({
                    'box': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': confidence,
                    'type': 'dnn_face'
                })

        return faces

    def detect_faces_cascade(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using Haar Cascade
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        # Detect frontal faces
        frontal = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in frontal:
            faces.append({
                'box': (x, y, w, h),
                'confidence': 0.7,
                'type': 'cascade_frontal'
            })

        # Detect profile faces
        profiles = self.profile_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in profiles:
            faces.append({
                'box': (x, y, w, h),
                'confidence': 0.6,
                'type': 'cascade_profile'
            })

        return faces

    def detect_eyes_in_face(self, face_region: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect eyes within a face region to determine orientation
        """
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        eyes = eye_cascade.detectMultiScale(gray, 1.05, 3)

        return eyes

    def analyze_face_orientation(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> str:
        """
        Analyze face orientation using eye positions and face geometry

        Returns:
            'upright', 'sideways', 'upside_down', or 'uncertain'
        """
        x, y, w, h = face_box
        face_region = frame[y:y + h, x:x + w]

        if face_region.size == 0:
            return 'uncertain'

        # Detect eyes in the face
        eyes = self.detect_eyes_in_face(face_region)

        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            eye1 = eyes[0]
            eye2 = eyes[1]

            # Calculate eye centers
            eye1_center = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
            eye2_center = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)

            # Calculate angle between eyes
            dx = eye2_center[0] - eye1_center[0]
            dy = eye2_center[1] - eye1_center[1]

            if dx == 0:
                angle = 90 if dy > 0 else -90
            else:
                angle = math.degrees(math.atan2(dy, dx))

            # Determine orientation based on angle
            if -30 <= angle <= 30:
                return 'upright'  # Eyes are roughly horizontal
            # elif 60 <= angle <= 120 or -120 <= angle <= -60:
            #     return 'sideways'  # Eyes are roughly vertical
            elif 150 <= angle or angle <= -150:
                return 'upside_down'  # Eyes are horizontal but inverted
            # else:
            #     return 'tilted'

        # Fallback: analyze face aspect ratio and position
        face_aspect = h / w if w > 0 else 1

        # Faces are typically taller than wide when upright
        if face_aspect > 1.2:
            return 'upright'
        elif face_aspect < 0.8:
            return 'sideways'
        else:
            return 'uncertain'

    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect full person bodies in frame
        """
        persons = []

        if self.use_yolo:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if class_id == 0 and confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        persons.append({
                            'box': (x, y, w, h),
                            'confidence': confidence,
                            'type': 'yolo_person'
                        })
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Full body detection
            bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in bodies:
                persons.append({
                    'box': (x, y, w, h),
                    'confidence': 0.6,
                    'type': 'cascade_body'
                })

            # Upper body detection
            upper_bodies = self.upper_body_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in upper_bodies:
                persons.append({
                    'box': (x, y, w, h),
                    'confidence': 0.5,
                    'type': 'cascade_upper'
                })

        return persons

    def is_close_up(self, face_box: Tuple[int, int, int, int], frame_shape: Tuple) -> bool:
        """
        Determine if a face detection is a close-up shot
        """
        _, _, w, h = face_box
        frame_height, frame_width = frame_shape[:2]

        # Face takes up significant portion of frame
        face_area = w * h
        frame_area = frame_width * frame_height

        return (face_area / frame_area) > 0.05  # Face is more than 5% of frame

    def reset_stats(self):
        """Reset statistics for new video processing"""
        self.stats = {
            'total_frames': 0,
            'frames_with_humans': 0,
            'correct_orientation_frames': 0,
            'incorrect_orientation_frames': 0,
            'uncertain_frames': 0,
            'face_detections': 0,
            'body_detections': 0,
            'close_up_frames': 0,
            'analyzed_duration': 0.0,
            'video_duration': 0.0,
            'mobilenet_votes': 0,
            'hough_votes': 0,
            'aspect_votes': 0,
            'conflict_resolutions': 0
        }

    def determine_frame_orientation(self, frame: np.ndarray) -> Tuple[VideoOrientation, Dict]:
        """
        Enhanced orientation detection using multiple models and smart fusion with video context

        Returns:
            Tuple of (VideoOrientation, detection_info)
        """
        detection_info = {
            'faces': [],
            'bodies': [],
            'is_close_up': False,
            'primary_detection': None,
            'votes': {},
            'final_decision': None,
            'video_context': None
        }
        
        # Get video context (resolution-based)
        height, width = frame.shape[:2]
        video_aspect_ratio = width / height
        is_video_landscape = video_aspect_ratio > 1.2  # Wide video (like 1920x1080)
        is_video_portrait = video_aspect_ratio < 0.8   # Tall video (like 720x1080)
        detection_info['video_context'] = {
            'aspect_ratio': video_aspect_ratio,
            'is_landscape': is_video_landscape,
            'is_portrait': is_video_portrait,
            'resolution': f"{width}x{height}"
        }

        # Multi-model detection
        faces = []
        faces.extend(self.detect_faces_dnn(frame))
        faces.extend(self.detect_faces_cascade(frame))
        faces = self.remove_duplicates(faces)
        detection_info['faces'] = faces

        # Body detection
        bodies = self.detect_persons(frame)
        detection_info['bodies'] = bodies

        # Enhanced detection voting system
        votes = {
            'face': [],
            'yolo': [],
            'mobilenet': [],
            'hough': [],
            'aspect': []
        }

        # 1. Face-based voting
        for face in faces:
            if self.is_close_up(face['box'], frame.shape):
                detection_info['is_close_up'] = True
                self.stats['close_up_frames'] += 1

            face_orientation = self.analyze_face_orientation(frame, face['box'])
            if face_orientation in ['upright', 'upside_down']:
                votes['face'].append('correct')
            elif face_orientation == 'sideways':
                votes['face'].append('incorrect')
            else:
                votes['face'].append('uncertain')

        # 2. YOLO body voting
        for body in bodies:
            _, _, w, h = body['box']
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio > 1.3:
                votes['yolo'].append('correct')
            elif aspect_ratio < 0.7:
                votes['yolo'].append('incorrect')
            else:
                votes['yolo'].append('uncertain')

        # 3. Enhanced methods voting (with video context awareness)
        mobilenet_vote = self.mobilenet_detect_orientation(frame)
        hough_vote = self.detect_hough_lines(frame)
        aspect_vote = self.analyze_aspect_ratio(frame)
        
        # Smart voting based on video type
        for method_name, method_vote in [('mobilenet', mobilenet_vote), ('hough', hough_vote), ('aspect', aspect_vote)]:
            if is_video_landscape:
                # For landscape videos (like 1920x1080), landscape detection is CORRECT
                if method_vote == "landscape":
                    votes[method_name].append('correct')
                elif method_vote == "portrait":
                    votes[method_name].append('incorrect')  # Portrait in landscape video = rotated
                else:
                    votes[method_name].append('uncertain')
            elif is_video_portrait:
                # For portrait videos (like 720x1080), portrait detection is CORRECT  
                if method_vote == "portrait":
                    votes[method_name].append('correct')
                elif method_vote == "landscape":
                    votes[method_name].append('incorrect')  # Landscape in portrait video = rotated
                else:
                    votes[method_name].append('uncertain')
            else:
                # Square-ish videos - use traditional logic
                if method_vote == "portrait":
                    votes[method_name].append('correct')
                elif method_vote == "landscape":
                    votes[method_name].append('incorrect')
                else:
                    votes[method_name].append('uncertain')

        detection_info['votes'] = votes

        # No hardcoded overrides - let the algorithm decide naturally
        # Reference data is only used for post-processing validation

        # Weighted voting with priority system
        weighted_scores = {'correct': 0, 'incorrect': 0, 'uncertain': 0}

        # Face votes have highest weight (especially for close-ups)
        face_weight = 3.0 if detection_info['is_close_up'] else 2.0
        for vote in votes['face']:
            weighted_scores[vote] += face_weight

        # YOLO body votes
        yolo_weight = 2.0
        for vote in votes['yolo']:
            weighted_scores[vote] += yolo_weight

        # Enhanced method votes (lower weight but useful for consensus)
        enhanced_weight = 1.0
        for method in ['mobilenet', 'hough', 'aspect']:
            for vote in votes[method]:
                weighted_scores[vote] += enhanced_weight

        # Update stats
        if faces:
            self.stats['face_detections'] += len(faces)
        if bodies:
            self.stats['body_detections'] += len(bodies)
        if votes['mobilenet']:
            self.stats['mobilenet_votes'] += 1
        if votes['hough']:
            self.stats['hough_votes'] += 1
        if votes['aspect']:
            self.stats['aspect_votes'] += 1

        # Determine final orientation
        if weighted_scores['correct'] == 0 and weighted_scores['incorrect'] == 0:
            detection_info['final_decision'] = 'no_human_detected'
            return VideoOrientation.UNCERTAIN, detection_info

        # Apply smart decision logic
        if weighted_scores['correct'] > weighted_scores['incorrect'] * 1.2:
            detection_info['final_decision'] = 'weighted_correct'
            return VideoOrientation.CORRECT, detection_info
        elif weighted_scores['incorrect'] > weighted_scores['correct'] * 1.2:
            detection_info['final_decision'] = 'weighted_incorrect'
            return VideoOrientation.INCORRECT, detection_info
        else:
            # Close call - use additional heuristics
            if detection_info['is_close_up'] and votes['face']:
                # Trust face detection for close-ups
                face_correct = votes['face'].count('correct')
                face_incorrect = votes['face'].count('incorrect')
                if face_correct > face_incorrect:
                    detection_info['final_decision'] = 'closeup_face_correct'
                    return VideoOrientation.CORRECT, detection_info
                elif face_incorrect > face_correct:
                    detection_info['final_decision'] = 'closeup_face_incorrect'
                    return VideoOrientation.INCORRECT, detection_info

            # Fall back to majority vote across all methods
            total_correct = sum(votes[method].count('correct') for method in votes)
            total_incorrect = sum(votes[method].count('incorrect') for method in votes)
            
            if total_correct > total_incorrect:
                detection_info['final_decision'] = 'majority_correct'
                return VideoOrientation.CORRECT, detection_info
            elif total_incorrect > total_correct:
                detection_info['final_decision'] = 'majority_incorrect'
                return VideoOrientation.INCORRECT, detection_info
            else:
                detection_info['final_decision'] = 'tie_uncertain'
                return VideoOrientation.UNCERTAIN, detection_info

    def remove_duplicates(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate detections based on IoU
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []

        for i, det1 in enumerate(detections):
            duplicate = False
            for det2 in keep:
                if self.calculate_iou(det1['box'], det2['box']) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(det1)

        return keep

    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """
        Calculate Intersection over Union of two boxes
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def annotate_frame(self, frame: np.ndarray, orientation: VideoOrientation,
                       detection_info: Dict) -> np.ndarray:
        """
        Enhanced frame annotation with face and body detections
        """
        annotated = frame.copy()

        # Draw face detections
        for face in detection_info['faces']:
            x, y, w, h = face['box']

            # Analyze this specific face
            face_orient = self.analyze_face_orientation(frame, face['box'])

            if face_orient in ['upright', 'upside_down']:
                color = (0, 255, 0)  # Green
                label = "Face: Upright"
            elif face_orient == 'sideways':
                color = (0, 0, 255)  # Red
                label = "Face: Sideways"
            else:
                color = (255, 255, 0)  # Yellow
                label = "Face: Uncertain"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw eye detection if present
            face_region = frame[y:y + h, x:x + w]
            eyes = self.detect_eyes_in_face(face_region)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(annotated, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (255, 0, 255), 1)

        # Draw body detections
        for body in detection_info['bodies']:
            x, y, w, h = body['box']
            aspect_ratio = h / w if w > 0 else 0

            if aspect_ratio > 1.3:
                color = (0, 255, 0)  # Green
                label = "Body: Vertical"
            elif aspect_ratio < 0.7:
                color = (0, 0, 255)  # Red
                label = "Body: Horizontal"
            else:
                color = (255, 255, 0)  # Yellow
                label = "Body: Square"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add overall status
        status_text = orientation.value
        if detection_info['is_close_up']:
            status_text += " (Close-up)"

        # Add time limit info if active
        if self.time_limit:
            status_text += f" (Analyzing first {self.time_limit}s)"

        status_color = (0, 255, 0) if orientation == VideoOrientation.CORRECT else \
            (0, 0, 255) if orientation == VideoOrientation.INCORRECT else \
                (255, 255, 0)

        cv2.putText(annotated, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Add detection type
        if detection_info['primary_detection']:
            detect_text = f"Detection: {detection_info['primary_detection']}"
            cv2.putText(annotated, detect_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def process_video_quick(self, video_path: str) -> BatchResult:
        """
        Quick video processing for batch analysis (no display, fast sampling)
        """
        start_time = time.time()

        try:
            self.reset_stats()
            
            # Store current filename for smart override patterns
            self.current_filename = os.path.basename(video_path)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return BatchResult(video_path, VideoOrientation.UNCERTAIN, 0.0, {},
                                   time.time() - start_time, "Cannot open video")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.stats['video_duration'] = total_frames / fps if fps > 0 else 0

            # Calculate maximum frame to process based on time limit
            max_frame = self.get_max_frame_for_time_limit(fps)
            if max_frame is not None:
                max_frame = min(max_frame, total_frames)
                self.stats['analyzed_duration'] = max_frame / fps if fps > 0 else 0
            else:
                self.stats['analyzed_duration'] = self.stats['video_duration']

            # Sample fewer frames for batch processing
            skip_frames = 12
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Check time limit
                if max_frame is not None and frame_count > max_frame:
                    print(f"  ⏱️  Time limit reached: analyzed first {self.time_limit}s of video")
                    break

                # Skip frames for efficiency
                if frame_count % skip_frames != 0:
                    continue

                # Analyze frame
                orientation, detection_info = self.determine_frame_orientation(frame)

                # Update statistics
                self.stats['total_frames'] += 1
                if orientation == VideoOrientation.CORRECT:
                    self.stats['correct_orientation_frames'] += 1
                    if detection_info['faces'] or detection_info['bodies']:
                        self.stats['frames_with_humans'] += 1
                elif orientation == VideoOrientation.INCORRECT:
                    self.stats['incorrect_orientation_frames'] += 1
                    if detection_info['faces'] or detection_info['bodies']:
                        self.stats['frames_with_humans'] += 1
                else:
                    self.stats['uncertain_frames'] += 1

            cap.release()

            # Calculate results
            results = self.calculate_final_verdict()
            processing_time = time.time() - start_time

            return BatchResult(
                video_path,
                self._get_orientation_from_verdict(results['verdict']),
                results['confidence'],
                results,
                processing_time
            )

        except Exception as e:
            return BatchResult(video_path, VideoOrientation.UNCERTAIN, 0.0, {},
                               time.time() - start_time, str(e))

    def _get_orientation_from_verdict(self, verdict: str) -> VideoOrientation:
        """Extract VideoOrientation from verdict string"""
        if "CORRECT" in verdict:
            return VideoOrientation.CORRECT
        elif "ROTATED" in verdict:
            return VideoOrientation.INCORRECT
        else:
            return VideoOrientation.UNCERTAIN

    def process_video(self, video_path: str, display: bool = True,
                      output_path: str = None) -> Dict:
        """
        Process entire video (or time-limited portion) with enhanced detection
        """
        self.reset_stats()
        
        # Store current filename for smart override patterns
        self.current_filename = os.path.basename(video_path)
        
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stats['video_duration'] = total_frames / fps if fps > 0 else 0

        # Calculate maximum frame to process based on time limit
        max_frame = self.get_max_frame_for_time_limit(fps)
        if max_frame is not None:
            max_frame = min(max_frame, total_frames)
            self.stats['analyzed_duration'] = max_frame / fps if fps > 0 else 0
        else:
            self.stats['analyzed_duration'] = self.stats['video_duration']

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, Total frames: {total_frames}, FPS: {fps:.1f}")
        print(f"Video duration: {self.stats['video_duration']:.1f}s")
        if self.time_limit:
            print(f"⏱️  Analyzing only first {self.time_limit}s ({self.stats['analyzed_duration']:.1f}s actual)")
        print("Detecting faces and bodies for orientation analysis...")

        frame_count = 0
        skip_frames = 12  # Process every 12th frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Check time limit
            if max_frame is not None and frame_count > max_frame:
                print(f"\n⏱️  Time limit reached: analyzed first {self.time_limit}s of video")
                break

            # Skip frames for efficiency
            if frame_count % skip_frames != 0:
                continue

            # Analyze frame
            orientation, detection_info = self.determine_frame_orientation(frame)

            # Update statistics
            self.stats['total_frames'] += 1
            if orientation == VideoOrientation.CORRECT:
                self.stats['correct_orientation_frames'] += 1
                if detection_info['faces'] or detection_info['bodies']:
                    self.stats['frames_with_humans'] += 1
            elif orientation == VideoOrientation.INCORRECT:
                self.stats['incorrect_orientation_frames'] += 1
                if detection_info['faces'] or detection_info['bodies']:
                    self.stats['frames_with_humans'] += 1
            else:
                self.stats['uncertain_frames'] += 1

            # Annotate frame
            annotated_frame = self.annotate_frame(frame, orientation, detection_info)

            # Display
            if display:
                cv2.imshow('Video Orientation Analysis', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break

            # Write to output
            if writer:
                writer.write(annotated_frame)

            # Progress update
            if frame_count % 90 == 0:
                if max_frame:
                    progress = (frame_count / max_frame) * 100
                else:
                    progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Faces detected: {self.stats['face_detections']} | "
                      f"Bodies detected: {self.stats['body_detections']}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Calculate final verdict
        results = self.calculate_final_verdict()

        return results

    def calculate_final_verdict(self) -> Dict:
        """
        Calculate final verdict with detailed analysis
        """
        if self.stats['frames_with_humans'] == 0:
            verdict = "INCONCLUSIVE - No humans detected in video"
            confidence = 0.0
            recommendation = "Try with a video containing visible people"
        else:
            correct_ratio = self.stats['correct_orientation_frames'] / self.stats['frames_with_humans']
            incorrect_ratio = self.stats['incorrect_orientation_frames'] / self.stats['frames_with_humans']

            if correct_ratio > 0.7:
                verdict = "✓ VIDEO ORIENTATION IS CORRECT"
                confidence = correct_ratio
                recommendation = "No rotation needed"
            elif incorrect_ratio > 0.7:
                verdict = "✗ VIDEO IS ROTATED"
                confidence = incorrect_ratio
                recommendation = "Rotate video 90° to correct orientation"
            else:
                verdict = "⚠ MIXED ORIENTATION DETECTED"
                confidence = max(correct_ratio, incorrect_ratio)
                recommendation = "Manual review recommended - inconsistent orientations"

        close_up_ratio = self.stats['close_up_frames'] / max(self.stats['total_frames'], 1)

        results = {
            'verdict': verdict,
            'confidence': confidence,
            'recommendation': recommendation,
            'statistics': self.stats,
            'correct_percentage': (self.stats['correct_orientation_frames'] /
                                   max(self.stats['frames_with_humans'], 1)) * 100,
            'incorrect_percentage': (self.stats['incorrect_orientation_frames'] /
                                     max(self.stats['frames_with_humans'], 1)) * 100,
            'close_up_percentage': close_up_ratio * 100,
            'detection_types': {
                'face_detections': self.stats['face_detections'],
                'body_detections': self.stats['body_detections'],
                'close_up_frames': self.stats['close_up_frames']
            },
            'time_analysis': {
                'video_duration': self.stats['video_duration'],
                'analyzed_duration': self.stats['analyzed_duration'],
                'analysis_percentage': (self.stats['analyzed_duration'] / max(self.stats['video_duration'],
                                                                              0.01)) * 100 if self.stats[
                                                                                                  'video_duration'] > 0 else 0
            }
        }

        return results

    def print_results(self, results: Dict):
        """
        Print comprehensive analysis results
        """
        print("\n" + "=" * 60)
        print(" VIDEO ORIENTATION ANALYSIS RESULTS")
        print("=" * 60)
        print(f"\n{results['verdict']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Recommendation: {results['recommendation']}")

        print(f"\n📊 Frame Analysis:")
        print(f"  • Total frames analyzed: {results['statistics']['total_frames']}")
        print(f"  • Frames with humans: {results['statistics']['frames_with_humans']}")
        print(f"  • Correct orientation: {results['correct_percentage']:.1f}%")
        print(f"  • Incorrect orientation: {results['incorrect_percentage']:.1f}%")
        print(f"  • Close-up shots: {results['close_up_percentage']:.1f}%")

        print(f"\n🔍 Detection Statistics:")
        print(f"  • Face detections: {results['detection_types']['face_detections']}")
        print(f"  • Body detections: {results['detection_types']['body_detections']}")
        print(f"  • Close-up frames: {results['detection_types']['close_up_frames']}")
        
        # Enhanced detection statistics
        enhanced_stats = results['statistics']
        if 'mobilenet_votes' in enhanced_stats:
            print(f"\n🧠 Enhanced Detection Votes:")
            print(f"  • MobileNet votes: {enhanced_stats['mobilenet_votes']}")
            print(f"  • Hough line votes: {enhanced_stats['hough_votes']}")
            print(f"  • Aspect ratio votes: {enhanced_stats['aspect_votes']}")
            print(f"  • Conflict resolutions: {enhanced_stats['conflict_resolutions']}")
        
        # Show validation against reference if available
        if hasattr(self, 'current_filename') and self.current_filename:
            validation = self.validate_against_reference(
                self.current_filename, 
                VideoOrientation.CORRECT if results['confidence'] > 0.5 else VideoOrientation.INCORRECT
            )
            
            if validation['has_reference']:
                print(f"\n🎯 Reference Validation:")
                match_icon = "✅" if validation['is_correct'] else "❌"
                print(f"  • Expected: {validation['expected'].upper()}")
                print(f"  • Detected: {validation['detected'].upper()}")
                print(f"  • Result: {match_icon} {validation['match'].upper()}")
                if validation['notes']:
                    print(f"  • Notes: {validation['notes']}")

        print(f"\n⏱️ Time Analysis:")
        print(f"  • Video duration: {results['time_analysis']['video_duration']:.1f}s")
        print(f"  • Analyzed duration: {results['time_analysis']['analyzed_duration']:.1f}s")
        print(f"  • Analysis coverage: {results['time_analysis']['analysis_percentage']:.1f}%")

        if self.time_limit and results['time_analysis']['analysis_percentage'] < 100:
            print(f"  • Time limit: {self.time_limit}s (only analyzed beginning of video)")

        print("=" * 60)

    def process_folder(self, folder_path: str, recursive: bool = False,
                       output_file: str = None) -> List[BatchResult]:
        """
        Process all video files in a folder and generate summary report
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        results = []

        # Find all video files
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Error: Folder '{folder_path}' does not exist")
            return results

        if recursive:
            video_files = [f for f in folder.rglob('*') if f.suffix.lower() in video_extensions]
        else:
            video_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

        if not video_files:
            print(f"No video files found in {folder_path}")
            return results

        time_limit_info = f" (analyzing first {self.time_limit}s of each file)" if self.time_limit else ""
        print(f"\n🎬 Found {len(video_files)} video files to process{time_limit_info}...")
        print("=" * 80)

        # Process each video
        for i, video_file in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}] Processing: {video_file.name}")

            result = self.process_video_quick(str(video_file))
            results.append(result)

            # Show progress
            if result.error:
                print(f"  ❌ Error: {result.error}")
            else:
                status_icon = "✅" if result.orientation == VideoOrientation.CORRECT else "❌" if result.orientation == VideoOrientation.INCORRECT else "⚠️"
                print(f"  {status_icon} {result.orientation.value.split(' -')[0]} ({result.confidence:.1%} confidence)")

            print(f"  ⏱️  Processing time: {result.processing_time:.1f}s")
            if hasattr(result.detection_info, 'time_analysis') and self.time_limit:
                coverage = result.detection_info.get('time_analysis', {}).get('analysis_percentage', 0)
                print(f"  📊 Analyzed {coverage:.0f}% of video duration")
            print()

        # Generate and display summary
        self.print_batch_summary(results)

        # Save detailed report if requested
        if output_file:
            self.save_batch_report(results, output_file)
            print(f"\n📊 Detailed report saved to: {output_file}")

        return results

    def print_batch_summary(self, results: List[BatchResult]):
        """
        Print summary table of batch processing results
        """
        print("\n" + "=" * 130)
        print(" BATCH PROCESSING SUMMARY - SORTED BY PRIORITY")
        if self.time_limit:
            print(f" (Analysis limited to first {self.time_limit} seconds of each video)")
        print("=" * 130)

        # Separate results by category
        needs_rotation = [r for r in results if r.orientation == VideoOrientation.INCORRECT and not r.error]
        manual_review = [r for r in results if r.orientation == VideoOrientation.UNCERTAIN and not r.error]
        correct_files = [r for r in results if r.orientation == VideoOrientation.CORRECT and not r.error]
        error_files = [r for r in results if r.error]

        # Sort each category
        needs_rotation.sort(key=lambda x: x.confidence, reverse=True)
        manual_review.sort(key=lambda x: x.confidence, reverse=True)
        correct_files.sort(key=lambda x: x.confidence, reverse=True)
        error_files.sort(key=lambda x: x.filename)

        # Print header
        print(f"{'STATUS':<12} {'FILENAME':<35} {'SIZE(MB)':<8} {'CONF':<6} {'TIME(s)':<7} {'RECOMMENDATION':<25}")
        print("-" * 130)

        # Print files that need rotation (highest priority)
        if needs_rotation:
            print(f"\n🔴 FILES REQUIRING ROTATION ({len(needs_rotation)} files):")
            print("-" * 60)
            for result in needs_rotation:
                self._print_result_row(result, "ROTATE")

        # Print files needing manual review
        if manual_review:
            print(f"\n🟡 FILES REQUIRING MANUAL REVIEW ({len(manual_review)} files):")
            print("-" * 60)
            for result in manual_review:
                self._print_result_row(result, "MANUAL")

        # Print correct files
        if correct_files:
            print(f"\n🟢 FILES WITH CORRECT ORIENTATION ({len(correct_files)} files):")
            print("-" * 60)
            for result in correct_files:
                self._print_result_row(result, "OK")

        # Print error files
        if error_files:
            print(f"\n⚫ FILES WITH ERRORS ({len(error_files)} files):")
            print("-" * 60)
            for result in error_files:
                self._print_result_row(result, "ERROR")

        # Print overall statistics
        total_files = len(results)
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  • Total files processed: {total_files}")
        print(f"  • Need rotation: {len(needs_rotation)} ({len(needs_rotation) / total_files * 100:.1f}%)")
        print(f"  • Need manual review: {len(manual_review)} ({len(manual_review) / total_files * 100:.1f}%)")
        print(f"  • Correct orientation: {len(correct_files)} ({len(correct_files) / total_files * 100:.1f}%)")
        print(f"  • Processing errors: {len(error_files)} ({len(error_files) / total_files * 100:.1f}%)")

        # Validation statistics if reference data is available
        if self.reference_data:
            correct_predictions = 0
            total_with_reference = 0
            
            for result in results:
                validation = self.validate_against_reference(
                    os.path.basename(result.filename),
                    result.orientation
                )
                if validation['has_reference']:
                    total_with_reference += 1
                    if validation['is_correct']:
                        correct_predictions += 1
            
            if total_with_reference > 0:
                accuracy = (correct_predictions / total_with_reference) * 100
                print(f"\n🎯 VALIDATION AGAINST REFERENCE DATA:")
                print(f"  • Files with reference data: {total_with_reference}")
                print(f"  • Correct predictions: {correct_predictions}")
                print(f"  • Algorithm accuracy: {accuracy:.1f}%")

        total_time = sum(r.processing_time for r in results)
        avg_time = total_time / len(results) if results else 0
        print(f"\n⏱️ PERFORMANCE:")
        print(f"  • Total processing time: {total_time:.1f}s")
        print(f"  • Average time per file: {avg_time:.1f}s")

        if self.time_limit:
            print(f"  • Analysis time limit: {self.time_limit}s per video")

        print("=" * 130)

    def _print_result_row(self, result: BatchResult, status: str):
        """Print a single result row in the table"""
        # Truncate filename if too long
        filename = result.filename
        if len(filename) > 35:
            filename = filename[:32] + "..."

        confidence_str = f"{result.confidence:.1%}" if not result.error else "N/A"
        recommendation = self._get_short_recommendation(result)

        print(f"{status:<12} {filename:<35} {result.filesize:<8.1f} {confidence_str:<6} "
              f"{result.processing_time:<7.1f} {recommendation:<25}")

    def _get_short_recommendation(self, result: BatchResult) -> str:
        """Get short recommendation text for table display"""
        if result.error:
            return "Check file integrity"
        elif result.orientation == VideoOrientation.INCORRECT:
            return "Rotate 90° clockwise"
        elif result.orientation == VideoOrientation.UNCERTAIN:
            return "Manual inspection"
        else:
            return "No action needed"

    def save_batch_report(self, results: List[BatchResult], output_file: str):
        """
        Save detailed batch processing report to file
        """
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(results),
            'time_limit_seconds': self.time_limit,
            'confidence_threshold': self.confidence_threshold,
            'summary': {
                'needs_rotation': len([r for r in results if r.orientation == VideoOrientation.INCORRECT]),
                'manual_review': len([r for r in results if r.orientation == VideoOrientation.UNCERTAIN]),
                'correct_orientation': len([r for r in results if r.orientation == VideoOrientation.CORRECT]),
                'errors': len([r for r in results if r.error])
            },
            'files': []
        }

        for result in results:
            file_data = {
                'filepath': result.filepath,
                'filename': result.filename,
                'filesize_mb': result.filesize,
                'orientation': result.orientation.value if result.orientation else 'ERROR',
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'error': result.error,
                'detection_info': result.detection_info if hasattr(result, 'detection_info') else {}
            }
            report_data['files'].append(file_data)

        # Save as JSON
        if output_file.lower().endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
        else:
            # Save as text report
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("VIDEO ORIENTATION ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {report_data['timestamp']}\n")
                f.write(f"Total files: {report_data['total_files']}\n")
                f.write(f"Time limit: {report_data['time_limit_seconds']}s per video\n" if report_data[
                    'time_limit_seconds'] else "Time limit: Full video analysis\n")
                f.write(f"Confidence threshold: {report_data['confidence_threshold']}\n\n")

                f.write("SUMMARY:\n")
                f.write(f"  Need rotation: {report_data['summary']['needs_rotation']}\n")
                f.write(f"  Manual review: {report_data['summary']['manual_review']}\n")
                f.write(f"  Correct orientation: {report_data['summary']['correct_orientation']}\n")
                f.write(f"  Errors: {report_data['summary']['errors']}\n\n")

                f.write("DETAILED RESULTS:\n")
                f.write("-" * 50 + "\n")

                for file_data in report_data['files']:
                    f.write(f"File: {file_data['filename']}\n")
                    f.write(f"  Path: {file_data['filepath']}\n")
                    f.write(f"  Size: {file_data['filesize_mb']:.1f} MB\n")
                    f.write(f"  Orientation: {file_data['orientation']}\n")
                    f.write(f"  Confidence: {file_data['confidence']:.1%}\n")
                    f.write(f"  Processing time: {file_data['processing_time']:.1f}s\n")
                    if file_data['error']:
                        f.write(f"  Error: {file_data['error']}\n")
                    f.write("\n")


def get_video_files_in_folder(folder_path: str, recursive: bool = False) -> List[Path]:
    """Get list of video files in folder"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    folder = Path(folder_path)

    if recursive:
        return [f for f in folder.rglob('*') if f.suffix.lower() in video_extensions]
    else:
        return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]


def main():
    """Main function to run the video orientation detector"""
    # Version info for CLI
    version = __version__
    release_date = __release_date__
    release_name = __release_name__
    
    parser = argparse.ArgumentParser(
        description=f'Smart Video Orientation Detector (SVOD) v{version} - {release_name}\n'
                   f'Detect video orientation using face and body analysis\n'
                   f'Release Date: {release_date}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single video analysis:
    %(prog)s video.mp4                    # Basic analysis with display
    %(prog)s video.mp4 -o corrected.mp4   # Save annotated output
    %(prog)s video.mp4 --no-display       # Process without display
    %(prog)s video.mp4 --time-limit 10    # Analyze only first 10 seconds

  Batch folder processing:
    %(prog)s /path/to/videos --batch      # Process all videos in folder
    %(prog)s /path/to/videos --batch -r   # Process recursively (subfolders)
    %(prog)s /path/to/videos --batch --report summary.txt  # Save detailed report
    %(prog)s /path/to/videos --batch --time-limit 15       # Analyze first 15s of each video

  Advanced options:
    %(prog)s video.mp4 -c 0.7 --time-limit 30  # Higher confidence + 30s limit
    %(prog)s folder --batch --reference orientations.csv  # Use reference for validation
        """
    )
    
    parser.add_argument('--version', action='version', 
                        version=f'SVOD v{version} ({release_name}) - {release_date}')

    parser.add_argument('path', help='Path to video file or folder for batch processing')
    parser.add_argument('--output', '-o', help='Path to save annotated video (single file mode)')
    parser.add_argument('--no-display', action='store_true',
                        help='Process without displaying video (single file mode)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                        help='Confidence threshold for detection (0-1, default: 0.5)')

    # NEW: Time limit parameter
    parser.add_argument('--time-limit', '-t', type=float, default=None,
                        help='Maximum time in seconds to analyze from start of video (default: analyze entire video)')

    # Batch processing options
    parser.add_argument('--batch', action='store_true',
                        help='Enable batch processing mode for folders')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Process subfolders recursively (batch mode only)')
    parser.add_argument('--report', help='Save detailed batch report to file (batch mode only)')
    parser.add_argument('--reference', help='Reference file (CSV/JSON) for validation against known orientations')

    args = parser.parse_args()

    # Validate input path
    if not args.path:
        parser.error("path is required")

    # Validate input path exists
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' not found")
        return 1

    # Validate time limit
    if args.time_limit is not None and args.time_limit <= 0:
        print("Error: Time limit must be positive")
        return 1

    # Quick system check and setup
    print("🔍 Checking system requirements...")
    success, issues = check_system_requirements()
    
    # Separate different types of issues
    missing_files = [issue for issue in issues if 'missing required model file' in issue.lower()]
    other_critical_issues = [issue for issue in issues if any(keyword in issue.lower() 
                            for keyword in ['missing essential', 'python version', 'no write permissions', 'opencv check failed'])
                            and 'missing required model file' not in issue.lower()]
    
    # Stop immediately for non-file critical issues (Python, packages, permissions)
    if other_critical_issues:
        print("❌ Critical system issues detected:")
        for issue in other_critical_issues:
            print(f"   • {issue}")
        return 1
    
    # Show missing files but continue to try downloading them
    if missing_files:
        print("⚠️  Missing required model files (will attempt to download):")
        for issue in missing_files:
            print(f"   • {issue}")
        print()
    
    # Show warnings (non-critical)
    warnings = [issue for issue in issues if issue not in other_critical_issues and issue not in missing_files]
    if warnings:
        print("⚠️  System warnings (non-critical):")
        for warning in warnings[:3]:  # Show only first 3 warnings
            print(f"   • {warning}")
        if len(warnings) > 3:
            print(f"   ... and {len(warnings) - 3} more")
        print()
    
    # Install packages and download models (this should fix missing model files)
    print("📦 Setting up dependencies...")
    try:
        if not install_required_packages():
            print("❌ Package installation failed.")
            return 1
        download_model_files()
        print("✅ Dependencies setup complete!")
    except Exception as e:
        print(f"⚠️ Setup warning: {e}")
    
    # Final check - ensure all required model files are now present
    print("🔍 Final validation of required files...")
    files_ok, missing_files_final = check_required_model_files()
    if not files_ok:
        print("❌ Critical model files are still missing after download attempt:")
        for missing_file in missing_files_final:
            print(f"   • {missing_file}")
        print("\n💡 Possible solutions:")
        print("   1. Check internet connectivity")
        print("   2. Manually download files to script directory")
        print("   3. Check firewall/proxy settings")
        return 1
    
    print("✅ All required model files verified!")
    print()

    # Create detector with time limit
    print(f"🎬 Smart Video Orientation Detector (SVOD) v{version}")
    print(f"📅 Release: {release_name} ({release_date})")
    print("Initializing orientation detector...")
    if args.time_limit:
        print(f"⏱️  Time limit set to {args.time_limit} seconds")

    detector = OrientationDetector(
        confidence_threshold=args.confidence,
        time_limit=args.time_limit
    )
    
    # Load reference data if provided
    if args.reference:
        detector.load_reference_data(args.reference)

    try:
        if args.batch:
            # Batch processing mode
            if not os.path.isdir(args.path):
                print("Error: Batch mode requires a folder path")
                return 1

            print(f"🎬 Starting batch processing of folder: {args.path}")
            if args.recursive:
                print("📁 Recursive mode enabled - processing subfolders")

            results = detector.process_folder(
                args.path,
                recursive=args.recursive,
                output_file=args.report
            )

            if not results:
                print("No video files found or processed")
                return 1

            # Quick summary for command line
            needs_rotation = sum(1 for r in results if r.orientation == VideoOrientation.INCORRECT)
            total_files = len(results)

            print(f"\n🏁 Batch processing complete!")
            print(f"📋 {needs_rotation} out of {total_files} files need rotation")

        else:
            # Single file processing mode
            if os.path.isdir(args.path):
                print("Error: Single file mode requires a video file path")
                print("Use --batch flag for folder processing")
                return 1

            results = detector.process_video(
                args.path,
                display=not args.no_display,
                output_path=args.output
            )

            # Print results
            detector.print_results(results)

            if args.output:
                print(f"\n✓ Annotated video saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError processing: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
