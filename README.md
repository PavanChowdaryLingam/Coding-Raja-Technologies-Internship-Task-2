# CODING RAJA TECHNOLOGIES!
## Artificial Intelligence Task 2: Object Detection and Tracking in Video

### Overview

This project demonstrates how to perform object detection and tracking in a video using artificial intelligence techniques. We utilize the YOLOv3 (You Only Look Once) deep learning model for object detection and OpenCV for real-time video processing and tracking.

### Requirements

- Python 3.x
- OpenCV (opencv-python)
- NumPy

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Objectives

- **Object Detection**: Utilize YOLOv3 to detect various objects in each frame of a video feed.
- **Object Tracking**: Implement tracking of detected objects across video frames to monitor their movements.
- **Visualization**: Draw bounding boxes around detected objects and label them for visual identification.

### Project Structure

- **`object_detection_tracking.py`**: Python script for object detection and tracking.
- **`yolov3.weights`**: Pre-trained weights for the YOLOv3 model.
- **`yolov3.cfg`**: Configuration file for the YOLOv3 model.
- **`coco.names`**: List of class names used in the COCO dataset for object labeling.

### Usage

1. **Download Model Files**:
   - Obtain `yolov3.weights` and `yolov3.cfg` from the official YOLO website or repository.
   - Download `coco.names` which contains the list of COCO dataset classes.

2. **Setup**:
   - Place `yolov3.weights`, `yolov3.cfg`, and `coco.names` in the same directory as `object_detection_tracking.py`.

3. **Run**:
   ```bash
   python object_detection_tracking.py
   ```

4. **Instructions**:
   - Press `Q` to exit the video feed.
   - Adjust detection parameters (thresholds, NMS) inside `object_detection_tracking.py` as needed.

---

This README provides a structured overview of the project, including setup instructions, usage guidelines, and project structure details. It focuses on providing clear instructions for users to set up and run the object detection and tracking application.
