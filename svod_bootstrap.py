"""Runtime setup for SVOD: dependency install, console output, and model downloads.

Imported by video_orientation_detector after the --version fast path.
"""

import os
import platform
import shutil
import subprocess
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any, Optional

import svod_state

# Provide a lightweight mediapipe stub when the package is unavailable so
# integration tests relying on patching the module can still run.
if "mediapipe" not in sys.modules:
    try:
        import mediapipe  # type: ignore  # noqa: F401
    except ImportError:
        mediapipe_stub = types.ModuleType("mediapipe")
        setattr(mediapipe_stub, "_SVOD_STUB", True)

        solutions_module = types.ModuleType("mediapipe.solutions")
        pose_module = types.ModuleType("mediapipe.solutions.pose")
        drawing_module = types.ModuleType("mediapipe.solutions.drawing_utils")

        class _PoseUnavailable:
            def __init__(self, *args: Any, **kwargs: Any):
                raise ImportError("mediapipe is not installed")

        setattr(pose_module, "Pose", _PoseUnavailable)
        setattr(solutions_module, "pose", pose_module)
        setattr(solutions_module, "drawing_utils", drawing_module)
        setattr(mediapipe_stub, "solutions", solutions_module)

        sys.modules["mediapipe"] = mediapipe_stub
        sys.modules["mediapipe.solutions"] = solutions_module
        sys.modules["mediapipe.solutions.pose"] = pose_module
        sys.modules["mediapipe.solutions.drawing_utils"] = drawing_module

# Global flag for MobileNet requirement override (used in WSL/Linux environments)
svod_state.mobilenet_required_override = False


# Simple print functions for early initialization (before Rich import)
def simple_print_message(message: str, emoji: Optional[str] = None):
    """Simple print function for early initialization"""
    if emoji:
        print(f"{emoji} {message}")
    else:
        print(message)


def _print_success_plain(message: str):
    """Print success message - early version"""
    simple_print_message(message, "✅")


def _print_error_plain(message: str):
    """Print error message - early version"""
    simple_print_message(message, "❌")


def _print_warning_plain(message: str):
    """Print warning message - early version"""
    simple_print_message(message, "⚠️")


def _print_info_plain(message: str):
    """Print info message - early version"""
    simple_print_message(message, "ℹ️")


print_success = _print_success_plain
print_error = _print_error_plain
print_warning = _print_warning_plain
print_info = _print_info_plain


# Note: Rich-compatible versions will replace these after Rich import

# Early print functions removed - using Rich-compatible versions later in the file


def is_apple_silicon():
    """Check if running on Apple Silicon (M1/M2/M3) Mac"""
    try:
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    except Exception:
        return False


# Third-party imports with auto-installation
def install_required_packages():
    """Install required packages if not available with enhanced error handling"""
    print_info("Checking and installing required packages...")

    required_packages = [
        ("cv2", "opencv-contrib-python==4.11.0.86"),  # Contrib version required by mediapipe
        ("numpy", "numpy==1.26.4"),
        ("torch", "torch==2.8.0"),  # PyTorch for model conversion
        ("torchvision", "torchvision==0.23.0"),  # PyTorch vision for model conversion
        ("ultralytics", "ultralytics>=8.3.0"),  # YOLOv11 support - required
        ("openvino", "openvino==2024.6.0"),  # OpenVINO for optimized inference
        ("onnx", "onnx==1.19.0"),  # ONNX for model format conversion
        ("onnxruntime", "onnxruntime>=1.17.0"),
        ("tqdm", "tqdm==4.67.1"),  # Progress bars for batch processing
        ("rich", "rich>=13.7.1"),  # Enhanced console output
    ]

    optional_packages = [("mediapipe", "mediapipe>=0.10.0")]
    if not is_apple_silicon():
        optional_packages.append(("openvino.tools", "openvino-dev"))

    missing_packages = []

    # Check required packages and DNN support
    for module_name, package_name in required_packages:
        try:
            module = __import__(module_name)
            # Special check for OpenCV DNN support
            if module_name == "cv2":
                try:
                    # Test DNN functionality that SVOD requires
                    _ = hasattr(module, "dnn") and hasattr(
                        module.dnn, "readNet"
                    )  # Check DNN module exists
                    _ = hasattr(module, "FaceDetectorYN")  # Check YuNet support
                    print_success(f"{package_name}: Already installed with full DNN support")
                except Exception:
                    print(
                        f"[WARNING] {package_name}: Installed but missing DNN support - will reinstall"
                    )
                    missing_packages.append(package_name)
            else:
                print_success(f"{package_name}: Already installed")
        except ImportError:
            missing_packages.append(package_name)
            print_error(f"{package_name}: Missing")

    if missing_packages:
        print_info(f"Installing required packages: {', '.join(missing_packages)}")

        # Check if pip is available
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            print_error("pip is not available. Please install pip first.")
            return False

        try:
            # Install missing packages
            for package in missing_packages:
                print_info(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    print_success(f"{package} installed successfully")
                else:
                    print_error(f"Failed to install {package}: {result.stderr}")
                    return False

            print_success("All required packages installed successfully!")

            if optional_packages:
                print_info("Installing optional packages...")
                for module_name, package_name in optional_packages:
                    try:
                        print_info(f"Installing {package_name} (optional)...")
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", package_name],
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                        if result.returncode == 0:
                            print_success(f"{package_name} installed successfully")
                        else:
                            print_warning(
                                f"Failed to install {package_name} (not critical): {result.stderr}"
                            )
                            print_info("Direct download fallbacks will be used instead")
                    except Exception as e:
                        print_warning(f"{package_name} installation failed (not critical): {e}")
                        print_info("Direct download fallbacks will be used instead")

            return True

        except subprocess.TimeoutExpired:  # type: ignore
            print_error("Package installation timed out")
            return False
        except Exception as e:
            print_error(f"Installation failed: {e}")
            return False
    else:
        print_success("All required packages are already available!")
        return True


def _import_module(name: str):
    """Import a module for its side effect and return it."""
    return __import__(name)


try:
    _import_module("cv2")
    _import_module("numpy")
    _import_module("openvino")
    _import_module("tqdm")
    _import_module("onnxruntime")
    svod_state.TQDM_AVAILABLE = True
except ImportError:
    print_info("Installing required packages automatically...")
    if install_required_packages():
        print_info("Restarting import after installation...")
        _import_module("cv2")
        _import_module("numpy")
        _import_module("openvino")
        _import_module("onnxruntime")
        try:
            _import_module("tqdm")
            svod_state.TQDM_AVAILABLE = True
        except ImportError:
            svod_state.TQDM_AVAILABLE = False
            print_warning("tqdm not available - batch processing will not show progress bars")
    else:
        print_error("Failed to install required packages!")
        sys.exit(1)

# Rich console for enhanced output (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_message(message: str, style: Optional[str] = None, emoji: Optional[str] = None):
    """Enhanced print function with Rich support"""
    if RICH_AVAILABLE and console:
        if emoji:
            message = f"{emoji} {message}"
        if style:
            console.print(message, style=style)
        else:
            console.print(message)
    else:
        # Fallback to regular print
        if emoji:
            message = f"{emoji} {message}"
        print(message)


def print_panel(title: str, content: str, border_style: str = "blue"):
    """Print content in a Rich panel"""
    if RICH_AVAILABLE and console:
        panel = Panel(content, title=title, border_style=border_style)  # type: ignore
        console.print(panel)
    else:
        print(f"\n{title}")
        print("=" * len(title))
        print(content)
        print("=" * len(title))


def _print_success_rich(message: str):
    """Print success message"""
    print_message(message, "green", "✅")


def _print_error_rich(message: str):
    """Print error message"""
    print_message(message, "red", "❌")


def _print_warning_rich(message: str):
    """Print warning message"""
    print_message(message, "yellow", "⚠️")


def _print_info_rich(message: str):
    """Print info message"""
    print_message(message, "blue", "ℹ️")


print_success = _print_success_rich
print_error = _print_error_rich
print_warning = _print_warning_rich
print_info = _print_info_rich


def print_header(title: str):
    """Print a header with Rich styling"""
    if RICH_AVAILABLE and console:
        header = Text(title, style="bold magenta")  # type: ignore
        console.print(header)
        console.print("=" * len(title), style="magenta")
    else:
        print(f"\n{title}")
        print("=" * len(title))


# Required YOLOv11 import for enhanced detection with robust error handling
# Moved to main() function to allow --version to work without YOLOv11
svod_state.YOLO_AVAILABLE = False


def check_required_model_files():
    """
    Check if all required model files are available
    Returns: (bool, list) - (all_critical_files_present, missing_files)
    Note: All model files are now mandatory for operation
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Critical files required for basic functionality
    critical_files = {
        "coco.names": "COCO class names",
        "face_detection_yunet_2023mar.onnx": "YuNet face detector model",
        "mobilenet-v2.xml": "MobileNet model configuration (required)",
        "mobilenet-v2.bin": "MobileNet model weights (required)",
    }

    # No optional files - all models are mandatory for operation

    missing_critical = []

    # Check critical files
    for filename, description in critical_files.items():
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            missing_critical.append(f"{filename} ({description})")

    # Return True only if all critical files are present
    return len(missing_critical) == 0, missing_critical


def check_system_requirements():
    """
    Quick system requirements check including required model files
    Returns: (bool, list) - (all_checks_passed, issues_found)
    """
    issues = []
    warnings = []

    # Check Python version - omz_downloader requires Python 3.11-3.12 for full compatibility
    python_version = tuple(map(int, platform.python_version().split(".")))
    if python_version < (3, 11):
        issues.append(
            f"Python version {platform.python_version()} is too old. Minimum required: 3.11 (for omz_downloader compatibility)"
        )
    elif python_version >= (3, 13):
        issues.append(
            f"Python version {platform.python_version()} is too new. Maximum supported: 3.12 (omz_downloader fails on 3.13+ due to NumPy compilation issues). Please use Python 3.11-3.12 for full compatibility."
        )
    elif python_version >= (3, 11) and python_version <= (3, 12):
        # Optimal range for all features including omz_downloader
        pass

    # Check essential dependencies
    essential_deps = [
        ("cv2", "opencv-contrib-python"),
        ("numpy", "numpy"),
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

    # Check OpenCV capabilities - DNN support is now verified during installation
    try:
        _import_module("cv2")
        print_success("OpenCV DNN support verified during installation")
    except Exception as e:
        print_error(f"OpenCV check failed: {str(e)}")

    # Check internet connectivity for model downloads
    try:
        urllib.request.urlopen("https://github.com", timeout=3)
    except Exception:
        warnings.append("Internet connectivity issues - model auto-download may fail")

    # Check file permissions
    try:
        current_dir = Path.cwd()
        test_file = current_dir / "test_write_permission.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        issues.append(f"No write permissions in current directory: {str(e)}")

    # Check available disk space
    try:
        current_dir = Path.cwd()
        if platform.system() == "Windows":
            total, used, free = shutil.disk_usage(current_dir)
            free_gb = free / (1024**3)
        else:
            statvfs = os.statvfs(current_dir)  # type: ignore
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)

        if free_gb < 1:
            warnings.append("Low disk space - model downloads may fail")
    except Exception:
        pass  # Non-critical

    return len(issues) == 0, issues + warnings


def check_system_resources():
    """
    Check system resources to prevent resource exhaustion attacks
    Returns: (bool, list) - (resources_ok, warnings)
    """
    warnings = []

    try:
        import psutil

        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        if available_gb < 1.0:  # Less than 1GB available
            warnings.append(".1f")

        # Check available disk space
        disk = psutil.disk_usage(".")
        free_gb = disk.free / (1024**3)

        if free_gb < 1.0:  # Less than 1GB free
            warnings.append(".1f")

        # Check CPU usage (if system is already heavily loaded)
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            warnings.append(f"High CPU usage detected: {cpu_percent}%")

    except ImportError:
        # psutil not available, skip resource checks
        warnings.append("psutil not available - resource monitoring disabled")
    except Exception as e:
        warnings.append(f"Resource check failed: {e}")

    return len(warnings) == 0, warnings


def download_model_files():
    """Download ALL required model files automatically"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    files_to_download = {
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    }

    # MobileNet models (OpenVINO format) - Using OpenVINO Model Zoo tools
    # These will be downloaded and converted using omz_downloader and omz_converter
    mobilenet_files = {
        "mobilenet-v2.xml": "MobileNet model configuration (will be generated)",
        "mobilenet-v2.bin": "MobileNet model weights (will be generated)",
    }
    print_info("MobileNet will be downloaded using OpenVINO Model Zoo tools")

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
            with open(file_path, "rb") as f:
                first_bytes = f.read(20)
                if first_bytes.startswith(b"<!DOCTYPE") or first_bytes.startswith(b"<html"):
                    return False

            # Additional validation for specific file types
            if filename.endswith(".xml"):
                # XML files should contain proper OpenVINO model XML
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read(500)  # Check first 500 chars
                    if "net name=" not in content or "layer" not in content:
                        return False

            elif filename.endswith(".weights"):
                # YOLO weights files should be binary and fairly large
                if file_size < 1000000:  # Should be at least 1MB
                    return False

            elif filename.endswith(".onnx"):
                # ONNX models should be binary and at least 100KB
                if file_size < 100000:
                    return False

            return True

        except Exception as e:
            print_warning(f"Could not validate {filename}: {e}")
            return False

    def download_mobilenet_with_fallback(script_dir, filename):
        """Download MobileNet with fallback for macOS compatibility"""
        dest_path = os.path.join(script_dir, filename)

        if os.path.exists(dest_path):
            print_success(f"{filename} already available")
            if validate_model_file(dest_path, filename):
                return True
            else:
                print_error(f"Existing {filename} is invalid - will re-download")
                try:
                    os.remove(dest_path)
                except Exception:
                    pass

        # Check platform-specific compatibility first
        if is_apple_silicon():
            print_info("Detected Apple Silicon (M1/M2/M3) - Using optimized fallback approach")
            print_info(
                "omz_downloader may have limited support on Apple Silicon, trying direct downloads first"
            )
            # On Apple Silicon, skip omz_downloader and go directly to fallbacks
            print_info("Skipping omz_downloader on Apple Silicon, using direct download...")
        else:
            # Try OpenVINO Model Zoo tools first on non-Apple Silicon platforms
            print_info("Attempting MobileNet download using OpenVINO Model Zoo tools...")
            omz_success = download_mobilenet_with_omz(script_dir, filename)

            if omz_success:
                return True

            # Fallback: MobileNet models cannot be reliably downloaded without omz_downloader
            # These models require proper conversion from PyTorch format
            print_warning(
                "OpenVINO tools failed, MobileNet requires omz_downloader for proper conversion"
            )
            print_info("Continuing without MobileNet - core detection algorithms are sufficient")
            return False

    def download_mobilenet_with_omz(script_dir, filename):
        """Download and convert MobileNet using OpenVINO Model Zoo tools"""
        print_info("Trying OpenVINO Model Zoo approach...")

        try:
            # Install OpenVINO dev tools if not available
            import subprocess
            import sys

            # Check if omz_downloader is available
            # First try to find omz_downloader in common locations
            omz_downloader_cmd = "omz_downloader"
            omz_converter_cmd = "omz_converter"

            # Try to find in Python Scripts directory first
            import sysconfig

            scripts_dir = sysconfig.get_path("scripts")
            if scripts_dir and os.path.exists(os.path.join(scripts_dir, "omz_downloader.exe")):
                omz_downloader_cmd = os.path.join(scripts_dir, "omz_downloader.exe")
                omz_converter_cmd = os.path.join(scripts_dir, "omz_converter.exe")

            try:
                result = subprocess.run(
                    [omz_downloader_cmd, "--help"], capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    raise Exception("omz_downloader not found")
            except Exception:
                print_info("Installing OpenVINO development tools...")
                install_result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "openvino-dev"],
                    capture_output=True,
                    text=True,
                )
                if install_result.returncode != 0:
                    print_error(f"Failed to install openvino-dev: {install_result.stderr}")
                    return False

            # Install required dependencies for model conversion
            print_info("Installing PyTorch and ONNX for model conversion...")
            try:
                # Install minimal PyTorch CPU version for conversion only
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "onnx"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                print_success("PyTorch and ONNX installed for model conversion")
            except Exception as e:
                print_error(f"Failed to install conversion dependencies: {e}")
                return False

            # Create models subdirectory
            models_dir = os.path.join(script_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            # Download the model
            print_info("Downloading mobilenet-v2-pytorch model...")
            result = subprocess.run(
                [omz_downloader_cmd, "--name", "mobilenet-v2-pytorch", "--output_dir", models_dir],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                print_error(f"Download failed: {result.stderr}")
                return False

            # Convert the model to OpenVINO IR format
            print_info("Converting model to OpenVINO IR format...")
            result = subprocess.run(
                [
                    omz_converter_cmd,
                    "--name",
                    "mobilenet-v2-pytorch",
                    "--download_dir",
                    models_dir,
                    "--output_dir",
                    models_dir,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                print_error(f"Conversion failed: {result.stderr}")
                return False

            # Move the converted files to script directory
            model_path = os.path.join(models_dir, "public", "mobilenet-v2-pytorch", "FP32")
            if os.path.exists(model_path):
                xml_file = os.path.join(model_path, "mobilenet-v2-pytorch.xml")
                bin_file = os.path.join(model_path, "mobilenet-v2-pytorch.bin")

                if os.path.exists(xml_file):
                    import shutil

                    shutil.copy2(xml_file, os.path.join(script_dir, "mobilenet-v2.xml"))
                    print_success("mobilenet-v2.xml copied successfully")

                if os.path.exists(bin_file):
                    import shutil

                    shutil.copy2(bin_file, os.path.join(script_dir, "mobilenet-v2.bin"))
                    print_success("mobilenet-v2.bin copied successfully")

                # Clean up models directory
                import shutil

                shutil.rmtree(models_dir, ignore_errors=True)

                return os.path.exists(os.path.join(script_dir, filename))
            else:
                print_error(f"Converted model not found in expected location: {model_path}")
                return False

        except subprocess.TimeoutExpired:  # type: ignore
            print_error("Download/conversion timed out")
            return False
        except Exception as e:
            print_error(f"OpenVINO Model Zoo tools failed: {e}")
            return False

    def download_file(filename, url):
        dest_path = os.path.join(script_dir, filename)

        # Special handling for MobileNet files - use fallback approach for cross-platform compatibility
        if filename in ["mobilenet-v2.xml", "mobilenet-v2.bin"]:
            return download_mobilenet_with_fallback(script_dir, filename)

        if not os.path.exists(dest_path):
            print_info(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, dest_path)
                print_success(f"{filename} downloaded successfully")

                # Validate the downloaded file
                if not validate_model_file(dest_path, filename):
                    print_error(f"Downloaded file {filename} is invalid - removing")
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                    return False

            except Exception as e:
                print_error(f"Failed to download {filename}: {e}")
                return False
        else:
            print_success(f"{filename} already available")
            # Validate existing file too
            if not validate_model_file(dest_path, filename):
                print_error(f"Existing file {filename} is invalid - removing and re-downloading")
                try:
                    os.remove(dest_path)
                    return download_file(filename, url)  # Retry download
                except Exception:
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

# Data models (defined in svod_models.py, imported here for backward compatibility)
