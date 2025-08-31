import cv2
import numpy as np
from enum import Enum
import argparse
from typing import Tuple, List, Dict, Optional
import os
import math


class VideoOrientation(Enum):
    """Enum for video orientation states"""
    CORRECT = "CORRECT - Humans are upright"
    INCORRECT = "INCORRECT - Humans are sideways/rotated"
    UNCERTAIN = "UNCERTAIN - Cannot determine orientation"


class OrientationDetector:
    """Enhanced class for detecting video orientation based on human features"""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the orientation detector

        Args:
            confidence_threshold: Minimum confidence for detection (0-1)
        """
        self.confidence_threshold = confidence_threshold

        # Initialize face detection (works for close-ups)
        self.setup_face_detection()

        # Initialize body/person detection
        self.setup_person_detection()

        # Initialize feature detection for orientation
        self.setup_feature_detection()

        # Statistics for the video
        self.stats = {
            'total_frames': 0,
            'frames_with_humans': 0,
            'correct_orientation_frames': 0,
            'incorrect_orientation_frames': 0,
            'uncertain_frames': 0,
            'face_detections': 0,
            'body_detections': 0,
            'close_up_frames': 0
        }

    def setup_face_detection(self):
        """Setup multiple face detection methods for robustness"""
        # Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

        # DNN-based face detection (more robust)
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        config_path = "deploy.prototxt"

        if os.path.exists(model_path) and os.path.exists(config_path):
            self.face_net = cv2.dnn.readNet(model_path, config_path)
            self.use_dnn_face = True
        else:
            print("DNN face model not found. Using Haar Cascade only.")
            self.use_dnn_face = False

    def setup_person_detection(self):
        """Setup person/body detection"""
        # YOLO for full person detection
        weights_path = "yolov4.weights"
        config_path = "yolov4.cfg"

        if os.path.exists(weights_path) and os.path.exists(config_path):
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.use_yolo = True
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        else:
            print("YOLO not found. Using Haar Cascades for body detection.")
            self.use_yolo = False
            self.body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            )
            self.upper_body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            )

    def setup_feature_detection(self):
        """Setup facial landmark detection for precise orientation"""
        # Check if cv2.face module is available (requires opencv-contrib-python)
        self.use_landmarks = False
        try:
            # Only try to use face module if it exists
            if hasattr(cv2, 'face'):
                landmark_model = "lbfmodel.yaml"  # OpenCV's facial landmark model
                if os.path.exists(landmark_model):
                    self.landmark_detector = cv2.face.createFacemarkLBF()
                    self.landmark_detector.loadModel(landmark_model)
                    self.use_landmarks = True
                    print("Facial landmark detection enabled.")
                else:
                    print("Landmark model not found. Using geometric analysis only.")
            else:
                print("OpenCV face module not available. Using geometric analysis only.")
                print("(Optional: Install opencv-contrib-python for enhanced features)")
        except Exception as e:
            print(f"Could not setup landmark detection: {e}")
            print("Using geometric analysis only.")
            self.use_landmarks = False

    def detect_faces_dnn(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using DNN method with orientation hints

        Returns:
            List of face detections with confidence and bounds
        """
        if not self.use_dnn_face:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append({
                    'box': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': confidence,
                    'type': 'dnn_face'
                })

        return faces

    def detect_faces_cascade(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using Haar Cascade
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        # Detect frontal faces
        frontal = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in frontal:
            faces.append({
                'box': (x, y, w, h),
                'confidence': 0.7,
                'type': 'cascade_frontal'
            })

        # Detect profile faces
        profiles = self.profile_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in profiles:
            faces.append({
                'box': (x, y, w, h),
                'confidence': 0.6,
                'type': 'cascade_profile'
            })

        return faces

    def detect_eyes_in_face(self, face_region: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect eyes within a face region to determine orientation
        """
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        eyes = eye_cascade.detectMultiScale(gray, 1.05, 3)

        return eyes

    def analyze_face_orientation(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> str:
        """
        Analyze face orientation using eye positions and face geometry

        Returns:
            'upright', 'sideways', 'upside_down', or 'uncertain'
        """
        x, y, w, h = face_box
        face_region = frame[y:y + h, x:x + w]

        if face_region.size == 0:
            return 'uncertain'

        # Detect eyes in the face
        eyes = self.detect_eyes_in_face(face_region)

        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            eye1 = eyes[0]
            eye2 = eyes[1]

            # Calculate eye centers
            eye1_center = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
            eye2_center = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)

            # Calculate angle between eyes
            dx = eye2_center[0] - eye1_center[0]
            dy = eye2_center[1] - eye1_center[1]

            if dx == 0:
                angle = 90 if dy > 0 else -90
            else:
                angle = math.degrees(math.atan2(dy, dx))

            # Determine orientation based on angle
            if -30 <= angle <= 30:
                return 'upright'  # Eyes are roughly horizontal
            elif 60 <= angle <= 120 or -120 <= angle <= -60:
                return 'sideways'  # Eyes are roughly vertical
            elif 150 <= angle or angle <= -150:
                return 'upside_down'  # Eyes are horizontal but inverted
            else:
                return 'tilted'

        # Fallback: analyze face aspect ratio and position
        face_aspect = h / w if w > 0 else 1

        # Faces are typically taller than wide when upright
        if face_aspect > 1.2:
            return 'upright'
        elif face_aspect < 0.8:
            return 'sideways'
        else:
            return 'uncertain'

    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect full person bodies in frame
        """
        persons = []

        if self.use_yolo:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if class_id == 0 and confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        persons.append({
                            'box': (x, y, w, h),
                            'confidence': confidence,
                            'type': 'yolo_person'
                        })
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Full body detection
            bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in bodies:
                persons.append({
                    'box': (x, y, w, h),
                    'confidence': 0.6,
                    'type': 'cascade_body'
                })

            # Upper body detection
            upper_bodies = self.upper_body_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in upper_bodies:
                persons.append({
                    'box': (x, y, w, h),
                    'confidence': 0.5,
                    'type': 'cascade_upper'
                })

        return persons

    def is_close_up(self, face_box: Tuple[int, int, int, int], frame_shape: Tuple) -> bool:
        """
        Determine if a face detection is a close-up shot
        """
        _, _, w, h = face_box
        frame_height, frame_width = frame_shape[:2]

        # Face takes up significant portion of frame
        face_area = w * h
        frame_area = frame_width * frame_height

        return (face_area / frame_area) > 0.05  # Face is more than 5% of frame

    def determine_frame_orientation(self, frame: np.ndarray) -> Tuple[VideoOrientation, Dict]:
        """
        Determine the orientation of a single frame with detailed analysis

        Returns:
            Tuple of (VideoOrientation, detection_info)
        """
        detection_info = {
            'faces': [],
            'bodies': [],
            'is_close_up': False,
            'primary_detection': None
        }

        # Detect faces (works for close-ups)
        faces = []
        faces.extend(self.detect_faces_dnn(frame))
        faces.extend(self.detect_faces_cascade(frame))

        # Remove duplicate detections
        faces = self.remove_duplicates(faces)
        detection_info['faces'] = faces

        # Detect full bodies
        bodies = self.detect_persons(frame)
        detection_info['bodies'] = bodies

        # Analyze orientations
        orientations = []

        # Prioritize face analysis for close-ups
        for face in faces:
            if self.is_close_up(face['box'], frame.shape):
                detection_info['is_close_up'] = True
                self.stats['close_up_frames'] += 1

            face_orientation = self.analyze_face_orientation(frame, face['box'])
            if face_orientation in ['upright', 'upside_down']:
                orientations.append('correct')
                detection_info['primary_detection'] = 'face_upright'
            elif face_orientation == 'sideways':
                orientations.append('incorrect')
                detection_info['primary_detection'] = 'face_sideways'
            else:
                orientations.append('uncertain')

        # If no clear face orientation, check body orientations
        if not orientations or all(o == 'uncertain' for o in orientations):
            for body in bodies:
                _, _, w, h = body['box']
                aspect_ratio = h / w if w > 0 else 0

                if aspect_ratio > 1.3:
                    orientations.append('correct')
                    detection_info['primary_detection'] = 'body_vertical'
                elif aspect_ratio < 0.7:
                    orientations.append('incorrect')
                    detection_info['primary_detection'] = 'body_horizontal'
                else:
                    orientations.append('uncertain')

        # Update statistics
        if faces:
            self.stats['face_detections'] += len(faces)
        if bodies:
            self.stats['body_detections'] += len(bodies)

        # Determine overall orientation
        if not orientations:
            return VideoOrientation.UNCERTAIN, detection_info

        correct_count = orientations.count('correct')
        incorrect_count = orientations.count('incorrect')

        if correct_count > incorrect_count:
            return VideoOrientation.CORRECT, detection_info
        elif incorrect_count > correct_count:
            return VideoOrientation.INCORRECT, detection_info
        else:
            return VideoOrientation.UNCERTAIN, detection_info

    def remove_duplicates(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate detections based on IoU
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []

        for i, det1 in enumerate(detections):
            duplicate = False
            for det2 in keep:
                if self.calculate_iou(det1['box'], det2['box']) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(det1)

        return keep

    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """
        Calculate Intersection over Union of two boxes
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def annotate_frame(self, frame: np.ndarray, orientation: VideoOrientation,
                       detection_info: Dict) -> np.ndarray:
        """
        Enhanced frame annotation with face and body detections
        """
        annotated = frame.copy()

        # Draw face detections
        for face in detection_info['faces']:
            x, y, w, h = face['box']

            # Analyze this specific face
            face_orient = self.analyze_face_orientation(frame, face['box'])

            if face_orient in ['upright', 'upside_down']:
                color = (0, 255, 0)  # Green
                label = "Face: Upright"
            elif face_orient == 'sideways':
                color = (0, 0, 255)  # Red
                label = "Face: Sideways"
            else:
                color = (255, 255, 0)  # Yellow
                label = "Face: Uncertain"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw eye detection if present
            face_region = frame[y:y + h, x:x + w]
            eyes = self.detect_eyes_in_face(face_region)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(annotated, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (255, 0, 255), 1)

        # Draw body detections
        for body in detection_info['bodies']:
            x, y, w, h = body['box']
            aspect_ratio = h / w if w > 0 else 0

            if aspect_ratio > 1.3:
                color = (0, 255, 0)  # Green
                label = "Body: Vertical"
            elif aspect_ratio < 0.7:
                color = (0, 0, 255)  # Red
                label = "Body: Horizontal"
            else:
                color = (255, 255, 0)  # Yellow
                label = "Body: Square"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add overall status
        status_text = orientation.value
        if detection_info['is_close_up']:
            status_text += " (Close-up)"

        status_color = (0, 255, 0) if orientation == VideoOrientation.CORRECT else \
            (0, 0, 255) if orientation == VideoOrientation.INCORRECT else \
                (255, 255, 0)

        cv2.putText(annotated, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Add detection type
        if detection_info['primary_detection']:
            detect_text = f"Detection: {detection_info['primary_detection']}"
            cv2.putText(annotated, detect_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def process_video(self, video_path: str, display: bool = True,
                      output_path: str = None) -> Dict:
        """
        Process entire video with enhanced detection
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, Total frames: {total_frames}, FPS: {fps}")
        print("Detecting faces and bodies for orientation analysis...")

        frame_count = 0
        skip_frames = 25  # Process every 25th frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for efficiency
            if frame_count % skip_frames != 0:
                continue

            # Analyze frame
            orientation, detection_info = self.determine_frame_orientation(frame)

            # Update statistics
            self.stats['total_frames'] += 1
            if orientation == VideoOrientation.CORRECT:
                self.stats['correct_orientation_frames'] += 1
                if detection_info['faces'] or detection_info['bodies']:
                    self.stats['frames_with_humans'] += 1
            elif orientation == VideoOrientation.INCORRECT:
                self.stats['incorrect_orientation_frames'] += 1
                if detection_info['faces'] or detection_info['bodies']:
                    self.stats['frames_with_humans'] += 1
            else:
                self.stats['uncertain_frames'] += 1

            # Annotate frame
            annotated_frame = self.annotate_frame(frame, orientation, detection_info)

            # Display
            if display:
                cv2.imshow('Video Orientation Analysis', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break

            # Write to output
            if writer:
                writer.write(annotated_frame)

            # Progress update
            if frame_count % 90 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Faces detected: {self.stats['face_detections']} | "
                      f"Bodies detected: {self.stats['body_detections']}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Calculate final verdict
        results = self.calculate_final_verdict()

        return results

    def calculate_final_verdict(self) -> Dict:
        """
        Calculate final verdict with detailed analysis
        """
        if self.stats['frames_with_humans'] == 0:
            verdict = "INCONCLUSIVE - No humans detected in video"
            confidence = 0.0
            recommendation = "Try with a video containing visible people"
        else:
            correct_ratio = self.stats['correct_orientation_frames'] / self.stats['frames_with_humans']
            incorrect_ratio = self.stats['incorrect_orientation_frames'] / self.stats['frames_with_humans']

            if correct_ratio > 0.7:
                verdict = "✓ VIDEO ORIENTATION IS CORRECT"
                confidence = correct_ratio
                recommendation = "No rotation needed"
            elif incorrect_ratio > 0.7:
                verdict = "✗ VIDEO IS ROTATED"
                confidence = incorrect_ratio
                recommendation = "Rotate video 90° to correct orientation"
            else:
                verdict = "⚠ MIXED ORIENTATION DETECTED"
                confidence = max(correct_ratio, incorrect_ratio)
                recommendation = "Manual review recommended - inconsistent orientations"

        close_up_ratio = self.stats['close_up_frames'] / max(self.stats['total_frames'], 1)

        results = {
            'verdict': verdict,
            'confidence': confidence,
            'recommendation': recommendation,
            'statistics': self.stats,
            'correct_percentage': (self.stats['correct_orientation_frames'] /
                                   max(self.stats['frames_with_humans'], 1)) * 100,
            'incorrect_percentage': (self.stats['incorrect_orientation_frames'] /
                                     max(self.stats['frames_with_humans'], 1)) * 100,
            'close_up_percentage': close_up_ratio * 100,
            'detection_types': {
                'face_detections': self.stats['face_detections'],
                'body_detections': self.stats['body_detections'],
                'close_up_frames': self.stats['close_up_frames']
            }
        }

        return results

    def print_results(self, results: Dict):
        """
        Print comprehensive analysis results
        """
        print("\n" + "=" * 60)
        print(" VIDEO ORIENTATION ANALYSIS RESULTS")
        print("=" * 60)
        print(f"\n{results['verdict']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Recommendation: {results['recommendation']}")

        print(f"\n📊 Frame Analysis:")
        print(f"  • Total frames analyzed: {results['statistics']['total_frames']}")
        print(f"  • Frames with humans: {results['statistics']['frames_with_humans']}")
        print(f"  • Correct orientation: {results['correct_percentage']:.1f}%")
        print(f"  • Incorrect orientation: {results['incorrect_percentage']:.1f}%")
        print(f"  • Close-up shots: {results['close_up_percentage']:.1f}%")

        print(f"\n🔍 Detection Statistics:")
        print(f"  • Face detections: {results['detection_types']['face_detections']}")
        print(f"  • Body detections: {results['detection_types']['body_detections']}")
        print(f"  • Close-up frames: {results['detection_types']['close_up_frames']}")
        print("=" * 60)


def main():
    """Main function to run the video orientation detector"""
    parser = argparse.ArgumentParser(
        description='Detect video orientation using face and body analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                    # Basic analysis with display
  %(prog)s video.mp4 -o corrected.mp4   # Save annotated output
  %(prog)s video.mp4 --no-display       # Process without display
  %(prog)s video.mp4 -c 0.7             # Higher confidence threshold
        """
    )

    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Path to save annotated video')
    parser.add_argument('--no-display', action='store_true',
                        help='Process without displaying video')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                        help='Confidence threshold for detection (0-1, default: 0.5)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        return 1

    # Create detector
    print("Initializing orientation detector...")
    detector = OrientationDetector(confidence_threshold=args.confidence)

    # Process video
    try:
        results = detector.process_video(
            args.video_path,
            display=not args.no_display,
            output_path=args.output
        )

        # Print results
        detector.print_results(results)

        if args.output:
            print(f"\n✓ Annotated video saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError processing video: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
