import time
import sys
import os

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_orientation_detector import OrientationDetector

print('=== PERFORMANCE COMPARISON: MediaPipe Impact ===')

# Test without MediaPipe (disable it)
print('\n--- Testing WITHOUT MediaPipe ---')
detector1 = OrientationDetector(time_limit=10)
detector1.mediapipe_available = False  # Disable MediaPipe

start_time = time.time()
result1 = detector1.process_video(r'C:\Users\boris\Videos\P2170127.mp4', display=False)
time1 = time.time() - start_time

print('.1f')
print(f'Pose detections: {detector1.stats.get("pose_detections", 0)}')

# Test with MediaPipe
print('\n--- Testing WITH MediaPipe ---')
detector2 = OrientationDetector(time_limit=10)

start_time = time.time()
result2 = detector2.process_video(r'C:\Users\boris\Videos\P2170127.mp4', display=False)
time2 = time.time() - start_time

print('.1f')
print(f'Pose detections: {detector2.stats.get("pose_detections", 0)}')

# Calculate difference
time_diff = time2 - time1
percent_increase = (time_diff / time1) * 100 if time1 > 0 else 0

print('\n=== PERFORMANCE ANALYSIS ===')
print('.1f')
print('.1f')
print('.1f')

if percent_increase < 20:
    print('✅ MINIMAL IMPACT: MediaPipe adds acceptable overhead')
elif percent_increase < 50:
    print('⚠️  MODERATE IMPACT: Consider for critical performance scenarios')
else:
    print('❌ SIGNIFICANT IMPACT: May need optimization or selective enabling')