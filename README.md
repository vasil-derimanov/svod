# Smart Video Orientation Detector

## Setup Requirements:
```
pip install opencv-python numpy
```
## Download yolov4.weights
```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

## Usage
```
# Basic usage
python video_orientation_detector.py path/to/video.mp4

# Process without display (faster)
python video_orientation_detector.py path/to/video.mp4 --no-display

# Adjust confidence threshold
python video_orientation_detector.py path/to/video.mp4 --confidence 0.7
```