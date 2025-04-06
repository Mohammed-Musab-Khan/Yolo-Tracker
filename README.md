# YOLOv8 Object Detection and Tracking System

![Computer Vision](https://img.shields.io/badge/Computer-Vision-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)

**Author**: Mohammed Musab Khan

A powerful compuster vision system for object detection and tracking using YOLOv8 models. This Python project demonstrates:
- Image object detection
- Real-time vehicle tracking
- Customizable probability thresholds
- Visual bounding box annotations

## Features
- 🖼️ Detect objects in images with confidence scores
- 🚗 Track vehicles in videos with persistent IDs
- 📊 Print detailed object information (class, coordinates, probability)
- 🎨 Visualize detections with bounding boxes and labels
- ⚙️ Configurable confidence thresholds

## Installation
1. Clone this repository
```bash
git clone https://github.com/Mohammed-Musab-Khan/Yolo-Tracker.git
cd Yolo-Tracker
```

2. Install requirements
```bash
pip install -r _requirements.txt
```

3. Download YOLOv8 models (included in repository):
- yolov8m.pt (medium model)
- yolov8n.pt (nano model)

## Usage
### Image Object Detection
```python
python main.py
```
- Uncomment `object_predict1()` or `object_predict2()` in main.py
- Place your image in project root as `cat_dot_detected.png`

### Vehicle Tracking
```python
python main.py
```
- Uncomment `vehicle_tracking()` or `vehicle_tracking_assignment()`
- Place your video in project root as `highway.mp4`
- Press 'q' to quit

## Requirements
- Python 3.8+
- OpenCV (`pip install opencv-python`)
- Ultralytics YOLOv8 (`pip install ultralytics`)
- See `_requirements.txt` for complete list

## Sample Output
![Object Detection Example](cat_dog.jpg)
*Example of detected objects in an image*

![Vehicle Tracking Example](highway.mp4)
*Example of vehicle tracking in a video*

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
