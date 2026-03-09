#!/usr/bin/env python3
"""
SVOD Project Cleanup Script
Version: 2.0.0
Last Updated: 2026-03-06
Follows rules from copilot-instructions.md for safe project cleanup
"""

import os
import shutil
import glob
from pathlib import Path


def print_colored(text, color="white"):
    """Print colored text (simplified for cross-platform compatibility)"""
    colors = {
        "cyan": "\033[96m",
        "yellow": "\033[93m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "gray": "\033[90m",
        "white": "\033[97m",
        "red": "\033[91m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def is_protected(path):
    """Check if path is protected according to copilot-instructions.md"""
    # Critical files that MUST NOT be deleted
    protected_files = [
        "video_orientation_detector.py",
        "reference_orientations.csv",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        "README.md",
        "LICENSE",
        "MANIFEST.in",
        "coco.names",
        "inspect_rotation.py",
        "cleanup.ps1",
        "cleanup.py",
        ".pre-commit-config.yaml",
        ".flake8",
        ".gitignore",
    ]

    # Critical folders that MUST NOT be deleted (including all contents)
    protected_folders = ["tests", "testing", ".vscode", ".github", "performance_baselines", "release"]

    # Normalize path separators
    path = path.replace("\\", "/")

    # Check exact file matches
    for file in protected_files:
        if path == file or path.endswith("/" + file) or path.endswith("\\" + file):
            return True

    # Check if path is inside protected folders
    for folder in protected_folders:
        if path.startswith(folder + "/") or path.startswith(folder + "\\") or path == folder:
            return True

    return False


def safe_remove(path, description=""):
    """Safely remove a file or directory with protection checks"""
    if not os.path.exists(path):
        print_colored(f"⚪ Not found: {path}", "gray")
        return False

    if is_protected(path):
        print_colored(f"🛡️  Protected: {path} {description}", "blue")
        return False

    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print_colored(f"✅ Removed: {path} {description}", "green")
        return True
    except Exception as e:
        print_colored(f"❌ Failed to remove {path}: {e}", "red")
        return False


def main():
    print_colored(
        "🧹 SVOD Cleanup Script v2.0.0 - Safe project cleanup following copilot-instructions.md",
        "cyan",
    )
    print_colored("⚠️  This script will only remove truly unnecessary files and folders", "yellow")
    print_colored("✅ All critical files and folders will be preserved", "green")

    removed_count = 0

    # Remove legacy model files (retain current YOLOv11 assets)
    unnecessary_model_files = [
        "yolov4.cfg",  # Old YOLOv4 config
        "yolov4.weights",  # Old YOLOv4 weights
        "res10_300x300_ssd_iter_140000.caffemodel",  # Old Caffe face detector
        "deploy.prototxt",  # Old Caffe config
        "lbfmodel.yaml",  # Old LBF landmark model
        "yolov8n.pt",  # Old YOLOv8 model
        "yolov10n.pt",  # Old YOLOv10 model
    ]

    print_colored("\n📂 Removing unnecessary model files...", "yellow")
    for file in unnecessary_model_files:
        if safe_remove(file, "(unnecessary model file)"):
            removed_count += 1

    # Remove old test virtual environments (keeping current test_env)
    old_test_envs = [
        ".venv-clean",
        ".venv-test",
        ".venv-wsl-clean",
        ".venv-test-linux",
        ".venv-test-linux-clean",
        ".venv-test-v492",
        ".venv-wsl-test-v492",
        ".venv-test-py313",
        ".venv-test-py311",
        ".venv-test-v410",
        ".venv-test-v410-windows",
        ".venv-test-v410-wsl",
        ".venv-final-test",
        ".venv-comprehensive-test",
        ".venv-accuracy-test",
        ".venv-rotation-test",
    ]

    print_colored("\n🗂️ Removing old test virtual environments...", "yellow")
    for env in old_test_envs:
        if safe_remove(env, "(old test environment)"):
            removed_count += 1

    # Remove temporary and cache files (but preserve protected folders)
    print_colored("\n🗑️ Removing temporary and cache files...", "yellow")

    # Remove __pycache__ directories (but not inside protected folders)
    for pycache in glob.glob("**/__pycache__", recursive=True):
        if not is_protected(pycache):
            if safe_remove(pycache, "(Python cache)"):
                removed_count += 1
        else:
            print_colored(f"�️  Protected: {pycache} (inside protected folder)", "blue")

    # Remove .pytest_cache directories
    for pytest_cache in glob.glob("**/.pytest_cache", recursive=True):
        if not is_protected(pytest_cache):
            if safe_remove(pytest_cache, "(pytest cache)"):
                removed_count += 1
        else:
            print_colored(f"🛡️  Protected: {pytest_cache} (inside protected folder)", "blue")

    # Remove coverage files
    coverage_files = ["htmlcov", ".coverage", "coverage.xml"]
    for cov_file in coverage_files:
        if os.path.exists(cov_file):
            if safe_remove(cov_file, "(coverage report)"):
                removed_count += 1

    # Remove .coverage.* files
    for cov_pattern in glob.glob(".coverage.*"):
        if safe_remove(cov_pattern, "(coverage data)"):
            removed_count += 1

    # Remove other temporary files
    temp_patterns = ["*.tmp", "*.temp", "*.log", ".project_status*"]
    for pattern in temp_patterns:
        for temp_file in glob.glob(pattern):
            if safe_remove(temp_file, "(temporary file)"):
                removed_count += 1

    # Check deployment files (don't auto-remove, require manual review)
    print_colored("\n🚀 Checking deployment files...", "yellow")
    deployment_files = ["Dockerfile", "docker-compose.yml"]
    for deploy_file in deployment_files:
        if os.path.exists(deploy_file):
            print_colored(
                f"⚠️  Review needed: {deploy_file} (marked for potential removal)", "yellow"
            )

    # Summary
    print_colored(f"\n✨ Safe cleanup completed! Removed {removed_count} items", "green")
    print_colored("🛡️  All critical files and folders have been preserved", "blue")

    print_colored("\n📋 Protected items:", "cyan")
    protected_files = [
        "video_orientation_detector.py",
        "reference_orientations.csv",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        "README.md",
        "LICENSE",
        "MANIFEST.in",
        "coco.names",
        "inspect_rotation.py",
        "cleanup.ps1",
        "cleanup.py",
        ".pre-commit-config.yaml",
        ".flake8",
        ".gitignore",
    ]
    protected_folders = ["tests", "testing", ".vscode", ".github", "performance_baselines", "release"]

    for file in protected_files:
        print_colored(f"   • {file}", "white")
    for folder in protected_folders:
        print_colored(f"   • {folder}/ (and all contents)", "white")

    print_colored("\n📂 Current project structure:", "cyan")
    try:
        items = sorted([f for f in os.listdir(".") if os.path.exists(f)])
        for item in items:
            if item in protected_files or item in protected_folders:
                print_colored(f"   • {item} 🛡️", "green")
            else:
                print_colored(f"   • {item}", "white")
    except Exception as e:
        print_colored(f"Could not list directory contents: {e}", "red")


if __name__ == "__main__":
    main()
