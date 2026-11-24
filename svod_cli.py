"""
SVOD Command-Line Interface

Contains the main CLI entry point and video file discovery utilities.
All heavy imports are deferred to avoid circular dependencies with
video_orientation_detector.py.
"""

import os
import argparse
import subprocess
import shutil
from typing import List, Optional
from pathlib import Path


def path_traversal_error(path: str) -> Optional[str]:
    """Return an error when a path component is '..' or a longer run of dots.

    A file name may contain the characters '..' (for example ``clip..final.mp4``).
    Only a component made entirely of dots is treated as traversal.
    """
    expanded = os.path.expanduser(path)
    for part in Path(expanded).parts:
        if len(part) >= 2 and set(part) == {"."}:
            return "Directory traversal not allowed in path"
    return None


def get_video_files_in_folder(folder_path: str, recursive: bool = False) -> List[Path]:
    """Get list of video files in folder"""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
    folder = Path(folder_path)

    if recursive:
        return [f for f in folder.rglob("*") if f.suffix.lower() in video_extensions]
    else:
        return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]


def auto_correct_video(
    video_path: str, rotation_direction: Optional[str], output_path: Optional[str] = None
) -> Optional[str]:
    """Rotate a video using ffmpeg based on detected rotation direction.

    Args:
        video_path: Path to the input video.
        rotation_direction: 'clockwise' or 'counterclockwise'.
        output_path: Explicit output path. If None, appends '_corrected' to stem.

    Returns:
        Path to the corrected video file, or None on failure.
    """
    if not shutil.which("ffmpeg"):
        print("[ERROR] ffmpeg not found on PATH — cannot auto-correct")
        return None

    if not rotation_direction or rotation_direction == "none":
        print("[WARNING] No rotation direction detected — skipping auto-correct")
        return None

    # ffmpeg transpose values: 1 = 90° CW, 2 = 90° CCW
    if rotation_direction == "clockwise":
        transpose = "1"
    else:
        transpose = "2"

    if output_path is None:
        stem = Path(video_path).stem
        suffix = Path(video_path).suffix or ".mp4"
        output_path = str(Path(video_path).parent / f"{stem}_corrected{suffix}")

    # Prevent overwriting the source file
    if os.path.abspath(output_path) == os.path.abspath(video_path):
        output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_corrected.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"transpose={transpose}",
        "-c:a",
        "copy",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            print(f"[ERROR] ffmpeg failed: {result.stderr[:500]}")
            return None
    except subprocess.TimeoutExpired:
        print("[ERROR] ffmpeg timed out (10 min limit)")
        return None
    except Exception as e:
        print(f"[ERROR] ffmpeg error: {e}")
        return None


def cli_main():
    """Main function to run the video orientation detector.

    This is the actual implementation. video_orientation_detector.main() delegates here.
    Imports are deferred to the function body to avoid circular imports.
    """
    # Deferred imports to avoid circular dependency with video_orientation_detector
    import video_orientation_detector as vod
    from svod_models import results_to_json, batch_results_to_json

    # Module-level symbols we need
    __version__ = vod.__version__
    __release_date__ = vod.__release_date__
    __release_name__ = vod.__release_name__
    print_success = vod.print_success
    print_error = vod.print_error
    print_warning = vod.print_warning
    print_info = vod.print_info
    print_header = vod.print_header
    check_system_requirements = vod.check_system_requirements
    check_required_model_files = vod.check_required_model_files
    check_system_resources = vod.check_system_resources
    install_required_packages = vod.install_required_packages
    download_model_files = vod.download_model_files
    is_apple_silicon = vod.is_apple_silicon
    OrientationDetector = vod.OrientationDetector
    VideoOrientation = vod.VideoOrientation

    # Python 3.13 UTF-8 encoding fix for Windows
    if os.name == "nt" and not os.environ.get("PYTHONIOENCODING"):
        os.environ["PYTHONIOENCODING"] = "utf-8"

    # Version info for CLI
    version = __version__
    release_date = __release_date__
    release_name = __release_name__

    parser = argparse.ArgumentParser(
        description=f"Smart Video Orientation Detector (SVOD) v{version} - {release_name}\n"
        f"Detect video orientation using face and body analysis\n"
        f"Release Date: {release_date}",
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
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"SVOD v{version} ({release_name}) - {release_date}"
    )

    parser.add_argument("path", help="Path to video file or folder for batch processing")
    parser.add_argument("--output", "-o", help="Path to save annotated video (single file mode)")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without displaying video (single file mode)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="Confidence threshold for detection (0-1, default: 0.5)",
    )

    # NEW: Time limit parameter
    parser.add_argument(
        "--time-limit",
        "-t",
        type=float,
        default=30.0,
        help="Maximum time in seconds to analyze from video (default: 30s, samples from start/middle/end)",
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="Analyze entire video without time limit"
    )
    parser.add_argument(
        "--quick-scan",
        action="store_true",
        help="Enable quick pre-scan to find object-rich segments for optimal analysis",
    )

    # Batch processing options
    parser.add_argument(
        "--batch", action="store_true", help="Enable batch processing mode for folders"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process subfolders recursively (batch mode only)",
    )
    parser.add_argument("--report", help="Save detailed batch report to file (batch mode only)")
    parser.add_argument(
        "--reference", help="Reference file (CSV/JSON) for validation against known orientations"
    )
    # Security hardening: Add batch processing limits
    parser.add_argument(
        "--max-files",
        type=int,
        default=1000,
        help="Maximum number of files to process in batch mode (default: 1000, max: 10000)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum directory depth for recursive processing (default: 10, max: 20)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout (suppresses normal display)",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=640,
        help="Max detection frame dimension in pixels (default: 640). Lower = faster but less accurate",
    )
    parser.add_argument(
        "--auto-correct",
        action="store_true",
        help="Automatically rotate INCORRECT videos using ffmpeg (saves to *_corrected.mp4)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export YOLO models to ONNX format for faster CPU inference, then exit",
    )
    args = parser.parse_args()

    # Handle --export-onnx early (does not require a video path)
    if args.export_onnx:
        try:
            from ultralytics import YOLO

            for pt_name in ["yolo11n.pt", "yolo11n-pose.pt"]:
                print(f"Exporting {pt_name} to ONNX...")
                model = YOLO(pt_name)
                export_path = model.export(format="onnx", imgsz=640, simplify=True)
                print(f"  Exported: {export_path}")
            print("ONNX export complete. Future runs will use ONNX models automatically.")
            return 0
        except Exception as e:
            print(f"ONNX export failed: {e}")
            return 1

    # Check YOLOv11 availability (moved here so --version works without it)
    try:
        # Simple check for ultralytics availability
        import importlib

        ultralytics_spec = importlib.util.find_spec("ultralytics")  # type: ignore
        if ultralytics_spec is not None:
            vod.YOLO_AVAILABLE = True
            print_success("YOLOv11 (ultralytics) detected - enhanced body detection enabled!")
        else:
            vod.YOLO_AVAILABLE = False
            print_error("YOLOv11 not available - YOLOv11 is required for operation")
            print_error("Please install ultralytics: pip install 'ultralytics>=8.3.0'")
            raise RuntimeError(
                "YOLOv11 is required for person detection. Please install: pip install 'ultralytics>=8.3.0'"
            )
    except Exception as e:
        vod.YOLO_AVAILABLE = False
        print_error(f"YOLOv11 check failed: {e}")
        print_error(
            "YOLOv11 is required for operation. Please install: pip install 'ultralytics>=8.3.0'"
        )
        raise RuntimeError(f"YOLOv11 is required for person detection. Installation failed: {e}")

    # Enhanced input validation with security checks
    print_info("Validating input parameters...")

    # Validate path is provided
    if not args.path:
        parser.error("path is required (use --help for usage information)")

    # Sanitize and validate path
    try:
        raw_path = args.path
        for char in ("\x00", "\n", "\r"):
            if char in raw_path:
                print_error(f"Invalid characters in path: {repr(char)}")
                return 1

        traversal_error = path_traversal_error(raw_path)
        if traversal_error:
            print_error(traversal_error)
            return 1

        # Convert to absolute path and resolve any symlinks
        args.path = os.path.abspath(os.path.expanduser(raw_path))

        # Check for path length limits (prevent extremely long paths)
        if len(args.path) > 4096:  # Common filesystem limit
            print_error(f"Path too long (max 4096 characters): {len(args.path)}")
            return 1

        # Check if path exists
        if not os.path.exists(args.path):
            print_error(f"Path does not exist: '{args.path}'")
            print_error("Please check the path and try again.")
            return 1

        # Check read permissions
        if not os.access(args.path, os.R_OK):
            print_error(f"No read permission for path: '{args.path}'")
            return 1

        # Additional validation for batch mode
        if args.batch:
            if not os.path.isdir(args.path):
                print_error(
                    f"Batch mode requires a directory, but '{args.path}' is not a directory"
                )
                print_error("For single file processing, remove the --batch flag")
                return 1

            # Check if directory is empty (warning only)
            try:
                if not os.listdir(args.path):
                    print_warning(f"Directory '{args.path}' appears to be empty")
            except PermissionError:
                print_error(f"Cannot read contents of directory: '{args.path}'")
                return 1

        else:
            # Single file mode validation
            if os.path.isdir(args.path):
                print_error(f"Single file mode requires a file, but '{args.path}' is a directory")
                print_error("For batch processing, add the --batch flag")
                return 1

            # Basic video file extension check (not foolproof but helpful)
            video_extensions = {
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".wmv",
                ".flv",
                ".webm",
                ".m4v",
                ".3gp",
            }
            file_ext = os.path.splitext(args.path)[1].lower()
            if file_ext not in video_extensions:
                print_warning(f"File extension '{file_ext}' may not be a supported video format")
                print_warning("Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP")
                print_warning("Proceeding anyway...")

    except (OSError, ValueError) as e:
        print_error(f"Invalid path format: {e}")
        return 1

    # Enhanced time limit validation
    if args.no_time_limit:
        args.time_limit = None
    elif args.time_limit is not None:
        if not isinstance(args.time_limit, (int, float)) or args.time_limit <= 0:
            print_error("Time limit must be a positive number (seconds)")
            return 1
        if args.time_limit > 3600:  # 1 hour limit
            print_error("Time limit too large (maximum 3600 seconds / 1 hour)")
            return 1
        if args.time_limit < 1:  # Minimum 1 second
            print_warning(
                f"Very short time limit ({args.time_limit}s) may not provide accurate results"
            )
            print_warning("Recommended minimum: 10 seconds for reliable detection")

    # Validate confidence threshold
    if not (0.0 <= args.confidence <= 1.0):
        print_error("Confidence threshold must be between 0.0 and 1.0")
        return 1

    # Validate output path if provided
    if args.output:
        try:
            output_dir = os.path.dirname(os.path.abspath(args.output))
            if output_dir and not os.path.exists(output_dir):
                print_error(f"Output directory does not exist: '{output_dir}'")
                return 1
            if not os.access(output_dir, os.W_OK):
                print_error(f"No write permission for output directory: '{output_dir}'")
                return 1
        except (OSError, ValueError) as e:
            print_error(f"Invalid output path: {e}")
            return 1

    # Validate reference file if provided
    if args.reference:
        try:
            ref_path = os.path.abspath(os.path.expanduser(args.reference))
            if not os.path.exists(ref_path):
                print_error(f"Reference file does not exist: '{ref_path}'")
                return 1
            if not os.access(ref_path, os.R_OK):
                print_error(f"Cannot read reference file: '{ref_path}'")
                return 1
            # Check file size (prevent extremely large files)
            if os.path.getsize(ref_path) > 10 * 1024 * 1024:  # 10MB limit
                print_error("Reference file too large (maximum 10MB)")
                return 1
        except (OSError, ValueError) as e:
            print_error(f"Invalid reference file path: {e}")
            return 1

    # Security hardening: Validate batch processing limits
    if args.max_files < 1 or args.max_files > 10000:
        print_error("max-files must be between 1 and 10000")
        return 1
    if args.max_depth < 1 or args.max_depth > 20:
        print_error("max-depth must be between 1 and 20")
        return 1

    # Security hardening: Additional path security checks
    try:
        # Enhanced path traversal protection
        import os.path

        # Resolve any symlinks and check for dangerous patterns
        resolved_path = os.path.realpath(args.path)
        if resolved_path != os.path.abspath(args.path):
            print_warning("Path contains symlinks - proceeding with resolved path")
            args.path = resolved_path

        # Check for suspicious path patterns that might indicate attacks
        suspicious_patterns = [
            "\\x",
            "\\u00",
            "\\u0000",
            "%2e%2e%2f",
            "%2e%2e/",
            "..%2f",
            "..\\",
            "<script",
            "javascript:",
            "data:",
            "vbscript:",
            "onload=",
            "onerror=",
        ]

        path_lower = args.path.lower()
        for pattern in suspicious_patterns:
            if pattern in path_lower:
                print_error(f"Suspicious path pattern detected: {pattern}")
                return 1

        # Check for excessive path depth (prevent deep recursion attacks)
        path_parts = args.path.replace("\\", "/").strip("/").split("/")
        if len(path_parts) > 50:  # Reasonable maximum path depth
            print_error("Path depth too deep (maximum 50 levels)")
            return 1

    except Exception as e:
        print_error(f"Path security validation failed: {e}")
        return 1

    print_success("Input validation passed!")

    # Quick system check and setup
    print_info("Checking system requirements...")
    success, issues = check_system_requirements()

    # Separate different types of issues
    missing_files = [issue for issue in issues if "missing required model file" in issue.lower()]
    other_critical_issues = [
        issue
        for issue in issues
        if any(
            keyword in issue.lower()
            for keyword in [
                "missing essential",
                "python version",
                "no write permissions",
                "opencv check failed",
            ]
        )
        and "missing required model file" not in issue.lower()
    ]

    # Stop immediately for non-file critical issues (Python, packages, permissions)
    if other_critical_issues:
        print_error("Critical system issues detected:")
        for issue in other_critical_issues:
            print_error(f"   • {issue}")
        return 1

    # Show missing files but continue to try downloading them
    if missing_files:
        print_warning("Missing required model files (will attempt to download):")
        for issue in missing_files:
            print_warning(f"   • {issue}")

    # Show warnings (non-critical)
    warnings = [
        issue
        for issue in issues
        if issue not in other_critical_issues and issue not in missing_files
    ]
    if warnings:
        print_warning("System warnings (non-critical):")
        for warning in warnings[:3]:  # Show only first 3 warnings
            print_warning(f"   • {warning}")
        if len(warnings) > 3:
            print_warning(f"   ... and {len(warnings) - 3} more")

    # Install packages and download models (this should fix missing model files)
    print_info("Setting up dependencies...")
    try:
        if not install_required_packages():
            print_error("Package installation failed.")
            return 1
        download_model_files()
        print_success("Dependencies setup complete!")
    except Exception as e:
        print_warning(f"Setup warning: {e}")

    # Final check - ensure critical model files are now present
    print_info("Final validation of required files...")
    files_ok, missing_files_final = check_required_model_files()

    if not files_ok:
        # Check if only MobileNet files are missing
        mobilenet_missing = [f for f in missing_files_final if "mobilenet" in f.lower()]
        other_missing = [f for f in missing_files_final if "mobilenet" not in f.lower()]

        if other_missing:
            print_error("Critical model files are still missing after download attempt:")
            for missing_file in other_missing:
                print_error(f"   • {missing_file}")
            print_error("\n[TIP] Possible solutions:")
            print_error("   1. Check internet connectivity")
            print_error("   2. Manually download files to script directory")
            print_error("   3. Check firewall/proxy settings")
            return 1
        elif mobilenet_missing:
            print_warning("MobileNet models could not be downloaded automatically:")
            for missing_file in mobilenet_missing:
                print_warning(f"   • {missing_file}")

            # Provide Apple Silicon specific guidance
            if is_apple_silicon():
                print_info(
                    "Apple Silicon (M1/M2/M3) detected - this is a known compatibility issue"
                )
                print_info("OpenVINO has limited support for Apple Silicon architecture")
                print_success(
                    "Core detection algorithms provide excellent accuracy without MobileNet"
                )
            else:
                print_info("This typically happens in some Linux/WSL environments")
                print_info("Core detection algorithms will still provide accurate results")

            # Temporarily disable MobileNet requirement for this run
            vod.mobilenet_required_override = False
    else:
        print_success("All model files verified!")

    # Security hardening: Check system resources
    print_info("Checking system resources...")
    resources_ok, resource_warnings = check_system_resources()

    if not resources_ok:
        print_warning("System resource issues detected:")
        for warning in resource_warnings:
            print_warning(f"   • {warning}")
        print_warning("Processing may be slow or unstable. Consider freeing up system resources.")

    # Security hardening: Safe defaults for processing parameters
    # Ensure time limit is reasonable to prevent excessive resource usage
    if args.time_limit is None:
        # Default time limit for security
        args.time_limit = 30.0
        print_info("Applied safe default: time_limit = 30.0 seconds")

    # Ensure confidence threshold is reasonable
    if args.confidence < 0.1:
        print_warning(
            f"Very low confidence threshold ({args.confidence}) may produce unreliable results"
        )
        print_info("Consider using confidence >= 0.3 for better reliability")

    # Create detector with time limit
    print_header(f"Smart Video Orientation Detector (SVOD) v{version}")
    print_info(f"Release: {release_name} ({release_date})")
    print_info("Initializing orientation detector...")
    if args.time_limit:
        print_info(f"Time limit set to {args.time_limit} seconds (analyzing first N seconds)")
    else:
        print_info("No time limit - analyzing entire video")

    detector = OrientationDetector(confidence_threshold=args.confidence, time_limit=args.time_limit)

    # Apply frame size override
    if args.frame_size != 640:
        detector._max_detect_dim = max(min(args.frame_size, 1920), 320)
        print_info(f"Detection frame size set to {detector._max_detect_dim}px")

    # Apply quick-scan mode
    if args.quick_scan:
        detector.quick_scan = True
        print_info("Quick pre-scan mode enabled — will probe for human-rich segments")

    # Load reference data if provided
    if args.reference:
        detector.load_reference_data(args.reference)

    try:
        if args.batch:
            # Batch processing mode (validation already done above)
            print_info(f"Starting batch processing of folder: {args.path}")
            if args.recursive:
                print_info("Recursive mode enabled - processing subfolders")

            # Security hardening: Apply batch processing limits
            results = detector.process_folder(
                args.path,
                recursive=args.recursive,
                output_file=args.report,
                max_files=args.max_files,
                max_depth=args.max_depth,
            )

            if not results:
                print_error("No video files found or processed")
                return 1

            # Quick summary for command line
            needs_rotation = sum(1 for r in results if r.orientation == VideoOrientation.INCORRECT)
            total_files = len(results)

            print_success("Batch processing complete!")
            print_info(f"{needs_rotation} out of {total_files} files need rotation")

            if args.json:
                print(batch_results_to_json(results))

            # Auto-correct batch results
            if args.auto_correct:
                corrected = 0
                for r in results:
                    if r.orientation == VideoOrientation.INCORRECT:
                        direction = (
                            r.detection_info.get("rotation_direction") if r.detection_info else None
                        )
                        out = auto_correct_video(r.filepath, direction)
                        if out:
                            print_success(f"Corrected: {r.filename} → {os.path.basename(out)}")
                            corrected += 1
                print_info(f"Auto-corrected {corrected}/{needs_rotation} videos")

        else:
            # Single file processing mode (validation already done above)
            print_info(f"Processing: {os.path.basename(args.path)}")
            results = detector.process_video(
                args.path, display=not args.no_display, output_path=args.output
            )

            if args.json:
                print(results_to_json(results))
            else:
                # Print results
                detector.print_results(results)

            if args.output:
                print_success(f"Annotated video saved to: {args.output}")

            # Auto-correct single file
            if args.auto_correct and results.get("orientation") == "INCORRECT":
                direction = results.get("rotation_direction")
                out = auto_correct_video(args.path, direction)
                if out:
                    print_success(f"Auto-corrected video saved to: {out}")

    except KeyboardInterrupt:
        print_warning("\n\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print_error(f"Error processing: {e}")
        return 1

    return 0
