#!/usr/bin/env python3
"""
Test the improved sideways portrait detection logic
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from video_orientation_detector import OrientationDetector

def test_sideways_portrait_detection():
    """Test the improved sideways portrait detection"""

    print("🧪 Testing Improved Sideways Portrait Detection")
    print("=" * 60)

    # Create detector
    detector = OrientationDetector(confidence_threshold=0.5, time_limit=30)

    # Test with reference orientations
    reference_file = "reference_orientations.csv"
    if os.path.exists(reference_file):
        print(f"📋 Loading reference data from {reference_file}")
        detector.load_reference_data(reference_file)

        # Test each reference video
        test_results = []
        for row in detector.reference_data.items():
            filename, ref_data = row
            expected = ref_data["expected"]
            notes = ref_data.get("notes", "")

            video_path = filename  # Assume videos are in current directory

            if os.path.exists(video_path):
                print(f"\n🎬 Testing: {filename}")
                print(f"Expected: {expected.upper()} ({notes})")

                try:
                    result = detector.process_video_quick(video_path)

                    detected = "incorrect" if result.orientation == detector.VideoOrientation.INCORRECT else "correct"
                    match = detected == expected.lower()

                    status_icon = "✅" if match else "❌"
                    print(f"{status_icon} Detected: {detected.upper()}")
                    print(f"Confidence: {result.confidence:.1%}")
                    print(f"Match: {'YES' if match else 'NO'}")

                    test_results.append({
                        'filename': filename,
                        'expected': expected,
                        'detected': detected,
                        'confidence': result.confidence,
                        'match': match,
                        'processing_time': result.processing_time
                    })

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
                    test_results.append({
                        'filename': filename,
                        'expected': expected,
                        'detected': 'error',
                        'confidence': 0.0,
                        'match': False,
                        'processing_time': 0.0
                    })
            else:
                print(f"⚠️  Video not found: {filename}")

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r['match'])
        accuracy = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Accuracy: {accuracy:.1f}%")

        # Show details for failed tests
        failed_tests = [r for r in test_results if not r['match']]
        if failed_tests:
            print(f"\n❌ Failed tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   {test['filename']}: Expected {test['expected']}, Detected {test['detected']}")

        return accuracy >= 80.0  # Consider test successful if accuracy >= 80%

    else:
        print(f"⚠️  Reference file not found: {reference_file}")
        print("Cannot run automated tests without reference data")
        return False

if __name__ == "__main__":
    success = test_sideways_portrait_detection()
    sys.exit(0 if success else 1)