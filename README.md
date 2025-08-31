# Smart Video Orientation Detector

## Setup Requirements:
```
pip install opencv-python numpy
```

```
pip install opencv-contrib-python
```

## Download yolov4.weights
```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

## Download res10_300x300_ssd_iter_140000.caffemodel
```
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

## Usage
```
# Basic usage
python video_orientation_detector.py path/to/video.mp4

# Process without display (faster)
python video_orientation_detector.py path/to/video.mp4 --no-display

# Process folder recursively with detailed report
python video_orientation_detector.py /path/to/videos --batch --recursive --report summary.txt

# Batch process folder, analyzing first 15 seconds of each video
python video_orientation_detector.py /path/to/videos --batch --time-limit 15

# Adjust confidence threshold
python video_orientation_detector.py path/to/video.mp4 --confidence 0.7
```