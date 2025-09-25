#!/usr/bin/env python3
"""
YOLOv10 Batch Test for Good_Examples and Bad_Examples folders
Compare YOLOv10 results with baseline performance data
"""

import os
import sys
import time
from pathlib import Path
import csv
from collections import Counter

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_orientation_detector import OrientationDetector


def load_reference_data():
    """Load reference orientation data from CSV"""
    reference_data = {}
    reference_file = "reference_orientations.csv"
    
    if not os.path.exists(reference_file):
        print(f"❌ Reference file not found: {reference_file}")
        return {}
    
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                expected = row['expected_orientation'].lower()
                confidence = row['confidence'].lower()
                notes = row.get('notes', '')
                
                reference_data[filename] = {
                    'expected': expected,
                    'confidence': confidence,
                    'notes': notes
                }
        
        print(f"✅ Loaded {len(reference_data)} reference entries from {reference_file}")
        return reference_data
        
    except Exception as e:
        print(f"❌ Error loading reference data: {e}")
        return {}


def test_folder_batch(folder_path: str, folder_name: str, max_files: int = None):
    """Test all videos in a folder with YOLOv10"""
    print(f"\n🧪 Testing {folder_name} folder: {folder_path}")
    print("=" * 60)
    
    # Get video files
    video_files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                video_files.append(os.path.join(folder_path, file))
    
    if not video_files:
        print(f"❌ No video files found in {folder_path}")
        return {}
    
    print(f"📹 Found {len(video_files)} video files")
    
    # Limit files if requested
    if max_files and len(video_files) > max_files:
        video_files = video_files[:max_files]
        print(f"📋 Testing first {max_files} files only")
    
    # Initialize detector with proper time limit (like baseline tests)
    print("🔧 Initializing YOLOv10 OrientationDetector...")
    # Create a fresh detector for each test to avoid state issues
    print("✅ YOLOv10 detector will be created fresh for each video!")
    
    # Test results
    results = {}
    total_time = 0
    
    # Load reference data
    reference_data = load_reference_data()
    
    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        print(f"\n📹 [{i}/{len(video_files)}] Testing: {filename}")
        
        try:
            # Create fresh detector for each video to avoid state issues
            print(f"   🔧 Creating fresh YOLOv10 detector for {filename}...")
            detector = OrientationDetector(confidence_threshold=0.5, time_limit=30.0)
            print(f"   ✅ Fresh detector ready!")
            
            start_time = time.time()
            
            # Process video
            video_results = detector.process_video(video_path, display=False)
            
            end_time = time.time()
            processing_time = end_time - start_time
            total_time += processing_time
            
            # Extract results
            orientation = video_results.get('orientation', 'UNCERTAIN')
            confidence = video_results.get('confidence', 0.0)
            rotation_angle = video_results.get('rotation_angle', 'N/A')
            method = video_results.get('method', 'unknown')
            
            # Compare with reference
            reference_match = "N/A"
            expected = "unknown"
            if filename in reference_data:
                expected = reference_data[filename]['expected']
                if orientation.lower() == expected.lower():
                    reference_match = "✅ MATCH"
                else:
                    reference_match = f"❌ MISMATCH (expected: {expected})"
            
            results[filename] = {
                'orientation': orientation,
                'confidence': confidence,
                'rotation_angle': rotation_angle,
                'method': method,
                'processing_time': processing_time,
                'expected': expected,
                'reference_match': reference_match
            }
            
            print(f"   Result: {orientation} (confidence: {confidence:.3f})")
            print(f"   Rotation: {rotation_angle}")
            print(f"   Method: {method}")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Reference: {reference_match}")
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            results[filename] = {
                'orientation': 'ERROR',
                'confidence': 0.0,
                'rotation_angle': 'N/A',
                'method': 'error',
                'processing_time': 0.0,
                'expected': 'unknown',
                'reference_match': f'❌ ERROR: {e}'
            }
    
    # Summary
    print(f"\n📊 {folder_name} Summary:")
    print("=" * 40)
    
    orientation_counts = Counter([r['orientation'] for r in results.values()])
    
    print(f"Total files: {len(results)}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per file: {total_time/len(results):.2f}s")
    
    print("\nOrientation Results:")
    for orientation, count in orientation_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {orientation}: {count} files ({percentage:.1f}%)")
    
    # Reference comparison
    matches = sum(1 for r in results.values() if r['reference_match'].startswith('✅'))
    total_with_reference = sum(1 for r in results.values() if r['expected'] != 'unknown')
    
    if total_with_reference > 0:
        match_percentage = (matches / total_with_reference) * 100
        print(f"\nReference Accuracy:")
        print(f"  Matches: {matches}/{total_with_reference} ({match_percentage:.1f}%)")
    
    return results


def compare_with_baseline():
    """Compare YOLOv10 results with baseline data"""
    print("\n🏆 YOLOv10 vs Baseline Comparison")
    print("=" * 60)
    
    # Expected baseline results (from performance_v4_19_0_baseline.txt)
    baseline_data = {
        'Bad_Examples': {
            'total_files': 13,
            'incorrect': 10,  # 76.9%
            'uncertain': 3,   # 23.1%
            'correct': 0      # 0%
        },
        'Good_Examples': {
            'total_files': 22,
            'correct': 20,    # 90.9%
            'false_negatives': 1,  # 4.5%
            'accuracy': 95    # >95%
        }
    }
    
    print("Baseline Data (v4.19.0):")
    print(f"  Bad_Examples: 76.9% INCORRECT, 23.1% UNCERTAIN")
    print(f"  Good_Examples: 90.9% CORRECT, accuracy >95%")
    
    return baseline_data


def main():
    """Main test function"""
    print("🚀 YOLOv10 Batch Test Suite")
    print("Testing Good_Examples and Bad_Examples folders")
    print("=" * 60)
    
    # Test folders
    good_examples_path = r"C:\Users\boris\Good_Examples"
    bad_examples_path = r"C:\Users\boris\Bad_Examples"
    
    # Test Good_Examples (limit to 10 files for speed)
    good_results = test_folder_batch(good_examples_path, "Good_Examples", max_files=10)
    
    # Test Bad_Examples (limit to 10 files for speed)
    bad_results = test_folder_batch(bad_examples_path, "Bad_Examples", max_files=10)
    
    # Compare with baseline
    baseline_data = compare_with_baseline()
    
    # Final summary
    print(f"\n🎯 Final YOLOv10 Test Results:")
    print("=" * 60)
    print(f"Good_Examples tested: {len(good_results)} files")
    print(f"Bad_Examples tested: {len(bad_results)} files")
    
    # Calculate accuracy for Good_Examples
    if good_results:
        good_correct = sum(1 for r in good_results.values() if r['orientation'] == 'CORRECT')
        good_accuracy = (good_correct / len(good_results)) * 100
        print(f"Good_Examples accuracy: {good_correct}/{len(good_results)} ({good_accuracy:.1f}%)")
    
    # Calculate results for Bad_Examples  
    if bad_results:
        bad_incorrect = sum(1 for r in bad_results.values() if r['orientation'] == 'INCORRECT')
        bad_uncertain = sum(1 for r in bad_results.values() if r['orientation'] == 'UNCERTAIN')
        bad_incorrect_pct = (bad_incorrect / len(bad_results)) * 100
        bad_uncertain_pct = (bad_uncertain / len(bad_results)) * 100
        print(f"Bad_Examples results: {bad_incorrect_pct:.1f}% INCORRECT, {bad_uncertain_pct:.1f}% UNCERTAIN")
    
    print("\n✅ YOLOv10 batch testing completed!")
    
    return good_results, bad_results


if __name__ == "__main__":
    main()