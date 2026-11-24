"""
Smart Video Orientation Detector (SVOD)
Enhanced video orientation detection using multi-model ensemble approach

Version: 4.28.0 - Project Structure Release
Date: November 2026
Author: Boris Jalov

The detector class lives in svod_detector. This module keeps the public import
path and the ``--version`` fast path.
"""

import sys
import types

__version__ = "4.28.0"
__release_date__ = "2026-11-24"
__release_name__ = "Project Structure Release"

# Fast path: allow `--version` to run without importing heavy dependencies
if __name__ == "__main__" and any(arg == "--version" for arg in sys.argv[1:]):
    print(f"SVOD v{__version__} ({__release_name__}) - {__release_date__}")
    sys.exit(0)

import svod_state  # noqa: E402


class _SVODModule(types.ModuleType):
    """Attribute access for flags that moved to svod_state."""

    _STATE_NAMES = ("YOLO_AVAILABLE", "mobilenet_required_override", "TQDM_AVAILABLE")

    def __getattr__(self, name):
        if name in self._STATE_NAMES:
            return getattr(svod_state, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in self._STATE_NAMES:
            setattr(svod_state, name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SVODModule

from svod_bootstrap import (  # noqa: E402
    check_required_model_files,
    check_system_requirements,
    check_system_resources,
    download_model_files,
    install_required_packages,
    is_apple_silicon,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from svod_cli import cli_main, get_video_files_in_folder  # noqa: E402
from svod_detector import OrientationDetector  # noqa: E402
from svod_models import BatchResult, VideoOrientation  # noqa: E402

__all__ = [
    "__version__",
    "__release_date__",
    "__release_name__",
    "BatchResult",
    "OrientationDetector",
    "VideoOrientation",
    "check_required_model_files",
    "check_system_requirements",
    "check_system_resources",
    "download_model_files",
    "get_video_files_in_folder",
    "install_required_packages",
    "is_apple_silicon",
    "main",
    "print_error",
    "print_header",
    "print_info",
    "print_success",
    "print_warning",
]


def main():
    """Main CLI entry point - delegates to svod_cli.cli_main()"""
    return cli_main()


if __name__ == "__main__":
    exit(main())
