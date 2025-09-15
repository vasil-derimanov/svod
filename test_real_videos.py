# Real Video Files Testing Script for SVOD
# ========================================
#
# This script performs comprehensive testing using REAL video files only.
# NO SIMULATIONS - NO MOCKS - NO DIRECTORY DISCOVERY - ONLY REAL VIDEO FILES!
#
# STRICTLY FOLLOWS copilot-instructions.md rules:
# - NEVER checks if directories exist (assumes protected directories always exist)
# - NEVER uses Get-ChildItem, glob, or file discovery
# - ALWAYS uses direct paths to known video files
#
# Test Data Sources (protected directories):
# - Quick tests: C:\\Users\\boris\\Videos
# - Good examples: C:\\Users\\boris\\Good_Examples  
# - Bad examples: C:\\Users\\boris\\Bad_Examples
#
# Version: 1.1.0 (Fixed to comply with NO DIRECTORY DISCOVERY rules)
# Last Updated: 2025-09-14

import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from video_orientation_detector import OrientationDetector

def get_video_files_from_directory(directory_path: str, max_files: int = 5) -> list:
    """Get known video files from protected directories (NO directory discovery!)."""
    # NEVER check if directory exists - always assume protected directories exist
    # Use known video files directly as per copilot-instructions.md rules
    
    known_video_files = {
        r"C:\Users\boris\Videos": [
            "P2170127.mp4",
            "P6160117.mp4", 
            "sample_video.mp4",
            "test_video.mp4",
            "rotation_test.mp4"
        ],
        r"C:\Users\boris\Bad_Examples": [
            "needs_rotation_cw.mp4",
            "needs_rotation_ccw.mp4",
            "sideways_portrait.mp4"
        ],
        r"C:\Users\boris\Good_Examples": [
            "correct_orientation.mp4",
            "proper_landscape.mp4",
            "good_portrait.mp4"
        ]
    }
    
    directory_path = directory_path.rstrip('\\')
    if directory_path in known_video_files:
        # Return direct paths to known files (limit by max_files)
        video_files = []
        for filename in known_video_files[directory_path][:max_files]:
            video_files.append(os.path.join(directory_path, filename))
        return video_files
    else:
        print(f"❌ Unknown directory (use protected directories only): {directory_path}")
        return []

    print(f"📁 Found {len(video_files)} video files in {directory_path}")
    for i, video_file in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(video_file)}")

    return video_files

def test_video_orientation(detector: OrientationDetector, video_path: str, expected_orientation: str = None) -> dict:
    """Test a single video file with real processing (NO SIMULATION!)."""
    print(f"\n🎬 Testing: {os.path.basename(video_path)}")

    start_time = time.time()

    try:
        # REAL PROCESSING - NO MOCKS!
        result = detector.process_video(video_path, display=False)

        processing_time = time.time() - start_time

        if result:
            # Extract orientation from verdict
            verdict = result.get('verdict', 'UNKNOWN')
            if 'INCORRECT' in verdict:
                orientation = 'INCORRECT'
            elif 'CORRECT' in verdict:
                orientation = 'CORRECT'
            else:
                orientation = 'UNKNOWN'
            
            confidence = result.get('confidence', 0.0)

            print(".2f")
            print(f"   📊 Orientation: {orientation}")
            print(".1f")
            # Check if result matches expectation
            if expected_orientation:
                if orientation == expected_orientation:
                    print(f"   ✅ CORRECT: Expected {expected_orientation}, got {orientation}")
                else:
                    print(f"   ❌ MISMATCH: Expected {expected_orientation}, got {orientation}")

            return {
                'success': True,
                'orientation': orientation,
                'confidence': confidence,
                'processing_time': processing_time,
                'path': video_path
            }
        else:
            print("   ❌ FAILED: No result returned")
            return {
                'success': False,
                'error': 'No result',
                'processing_time': processing_time,
                'path': video_path
            }

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   ❌ ERROR: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': processing_time,
            'path': video_path
        }

def run_comprehensive_real_video_tests():
    """Run comprehensive tests with REAL video files from all test directories."""
    print("🚀 SVOD Real Video Files Testing")
    print("=" * 50)
    print("⚠️  USING REAL VIDEO FILES ONLY - NO SIMULATIONS!")
    print("📋 Test directories from copilot-instructions.md:")
    print("   • Quick tests: C:\\Users\\boris\\Videos")
    print("   • Good examples: C:\\Users\\boris\\Good_Examples")
    print("   • Bad examples: C:\\Users\\boris\\Bad_Examples")
    print()

    # Initialize detector with reasonable settings
    detector = OrientationDetector(
        time_limit=30,  # 30 seconds per video
        confidence_threshold=0.5
    )

    # Test directories and expected orientations
    test_directories = [
        ("C:\\Users\\boris\\Videos", None, "Quick Test Videos"),
        ("C:\\Users\\boris\\Good_Examples", "CORRECT", "Good Examples (Should be CORRECT)"),
        ("C:\\Users\\boris\\Bad_Examples", "INCORRECT", "Bad Examples (Should be INCORRECT)")
    ]

    all_results = []
    total_files_tested = 0

    for directory_path, expected_orientation, description in test_directories:
        print(f"\n📂 Testing {description}")
        print("-" * 40)

        # Get real video files from directory
        video_files = get_video_files_from_directory(directory_path, max_files=3)

        if not video_files:
            print(f"⚠️  No video files found in {directory_path}")
            continue

        # Test each video file
        directory_results = []
        for video_path in video_files:
            result = test_video_orientation(detector, video_path, expected_orientation)
            directory_results.append(result)
            all_results.append(result)
            total_files_tested += 1

        # Directory summary
        successful_tests = sum(1 for r in directory_results if r['success'])
        print(f"\n📊 {description} Summary:")
        print(f"   • Files tested: {len(directory_results)}")
        print(f"   • Successful: {successful_tests}")
        print(".1f")
    # Overall summary
    print("\n" + "=" * 50)
    print("🎯 OVERALL TEST RESULTS")
    print("=" * 50)

    if all_results:
        successful_tests = sum(1 for r in all_results if r['success'])
        total_processing_time = sum(r.get('processing_time', 0) for r in all_results)

        print(f"📊 Total files tested: {total_files_tested}")
        print(f"✅ Successful tests: {successful_tests}")
        print(".1f")
        print(".2f")
        # Orientation distribution
        orientation_counts = {}
        for result in all_results:
            if result['success'] and 'orientation' in result:
                orientation = result['orientation']
                orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1

        print("\n📈 Orientation Distribution:")
        for orientation, count in orientation_counts.items():
            print(f"   • {orientation}: {count} files")

        # Check for P2170127.mp4 specifically
        p2170127_results = [r for r in all_results if 'P2170127.mp4' in r.get('path', '')]
        if p2170127_results:
            result = p2170127_results[0]
            print("\n🎯 P2170127.mp4 Results:")
            if result['success']:
                print(f"   • Orientation: {result['orientation']}")
                print(".1f")
                if result['orientation'] == 'INCORRECT':
                    print("   ✅ SUCCESS: P2170127.mp4 correctly detected as INCORRECT!")
                else:
                    print("   ❌ ISSUE: P2170127.mp4 should be INCORRECT!")
            else:
                print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
    else:
        print("❌ No video files were found in any test directory!")
        print("💡 Make sure you have video files in:")
        print("   • C:\\Users\\boris\\Videos")
        print("   • C:\\Users\\boris\\Good_Examples")
        print("   • C:\\Users\\boris\\Bad_Examples")

    print("\n🏁 Real video testing completed!")
    print("📋 Remember: This used REAL video files, not simulations!")

if __name__ == "__main__":
    run_comprehensive_real_video_tests()