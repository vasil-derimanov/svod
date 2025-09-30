import os
import sys

sys.path.append(".")
from video_orientation_detector import OrientationDetector

# Test P2170127.mp4 specifically
detector = OrientationDetector(time_limit=30)
result = detector.process_video(r"C:\Users\boris\Videos\P2170127.mp4", display=False)

if result:
    verdict = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    print(f"Verdict: {verdict}")
    print(f"Confidence: {confidence:.1%}")
    if "INCORRECT" in verdict:
        print("SUCCESS: P2170127.mp4 correctly detected as INCORRECT!")
    else:
        print("ISSUE: P2170127.mp4 should be INCORRECT!")
else:
    print("No result returned")
