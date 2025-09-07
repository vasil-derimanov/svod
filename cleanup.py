#!/usr/bin/env python3
"""
SVOD Project Cleanup Script
Removes all model files and test environments for vanilla testing
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
        "gray": "\033[90m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def main():
    print_colored("🧹 SVOD Cleanup Script - Preparing for vanilla testing", "cyan")
    
    # Model files to remove
    model_files = [
        "coco.names",
        "deploy.prototxt", 
        "lbfmodel.yaml",
        "mobilenet-v2.bin",
        "mobilenet-v2.xml",
        "res10_300x300_ssd_iter_140000.caffemodel",
        "yolov4.cfg",
        "yolov4.weights"
    ]
    
    print_colored("\n📂 Removing model files...", "yellow")
    for file in model_files:
        if os.path.exists(file):
            os.remove(file)
            print_colored(f"✅ Removed: {file}", "green")
        else:
            print_colored(f"⚪ Not found: {file}", "gray")
    
    # Test virtual environments to remove
    test_envs = [
        ".venv-clean",
        ".venv-test", 
        ".venv-wsl-clean",
        ".venv-test-linux",
        ".venv-test-linux-clean"
    ]
    
    print_colored("\n🗂️ Removing test virtual environments...", "yellow")
    for env in test_envs:
        if os.path.exists(env):
            shutil.rmtree(env)
            print_colored(f"✅ Removed: {env}", "green")
        else:
            print_colored(f"⚪ Not found: {env}", "gray")
    
    # Remove temp/cache files
    print_colored("\n🗑️ Removing temporary files...", "yellow")
    
    # Remove __pycache__ directories
    for pycache in glob.glob("**/__pycache__", recursive=True):
        shutil.rmtree(pycache)
        print_colored(f"✅ Removed: {pycache}", "green")
    
    # Remove .pyc files
    pyc_files = glob.glob("**/*.pyc", recursive=True)
    for pyc in pyc_files:
        os.remove(pyc)
        print_colored(f"✅ Removed: {pyc}", "green")
    
    # Remove temp files
    temp_patterns = ["*.tmp", "*.temp"]
    for pattern in temp_patterns:
        for temp_file in glob.glob(pattern):
            os.remove(temp_file)
            print_colored(f"✅ Removed: {temp_file}", "green")
    
    # Remove status file
    if os.path.exists(".project_status"):
        os.remove(".project_status")
        print_colored("✅ Removed: .project_status", "green")
    
    # Remove models directory if it exists
    if os.path.exists("models"):
        shutil.rmtree("models")
        print_colored("✅ Removed: models directory", "green")
    
    print_colored("\n✨ Cleanup completed! Ready for vanilla testing.", "green")
    print_colored("📋 Project now contains only:", "cyan")
    
    # List remaining files
    remaining_files = sorted([f for f in os.listdir('.') if not f.startswith('.')])
    for file in remaining_files:
        print_colored(f"   • {file}", "white")

if __name__ == "__main__":
    main()