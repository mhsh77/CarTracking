# Car Tracking with YOLOv8 Nano

This repository provides a solution for real-time car tracking using **YOLOv8 Nano**, an optimized lightweight version of the YOLO object detection algorithm. The project leverages advanced computer vision techniques to detect and track cars, making it ideal for applications requiring low latency and minimal resource consumption.


https://github.com/user-attachments/assets/d096aa5c-b512-4620-aaa0-bdc194fd86c2


---

## Features
- Real-time car detection and tracking.
- Lightweight implementation with YOLOv8 Nano for high-speed performance.
- Utilizes `opencv`, `cvzone`, and `sort` libraries for efficient video processing and object tracking.
- Configurable parameters for different input sources and detection thresholds.
- Demonstrates integration of a masking template to enhance detection accuracy in specific areas of interest.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Git

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/car-tracking-yolo8nano.git
   cd car-tracking-yolo8nano
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 Nano model weights:
   - You can download the pre-trained YOLOv8 Nano weights from [Ultralytics](https://github.com/ultralytics/ultralytics) or provide your own custom-trained weights.
   - Save the weights in the `weights/` directory.

---

## Code Explanation

### Key Components

#### YOLOv8 Nano
YOLOv8 Nano is a compact, efficient variant of the YOLO object detection framework. It is optimized for resource-constrained environments like edge devices while maintaining competitive accuracy and speed.

#### Tracker Integration
The project employs the **SORT** (Simple Online and Realtime Tracking) algorithm for robust object tracking. This ensures seamless tracking of detected cars across video frames.

#### Masking and Templates
Custom masks and templates are used to filter irrelevant areas in the video feed, focusing the detection on regions of interest and enhancing the overall accuracy.

### Code Walkthrough

1. **Model Initialization**
   ```python
   model = YOLO('yolov8n.pt')
   ```
   The YOLOv8 Nano model is loaded to perform object detection.

2. **Video Input and Masking**
   ```python
   cap = cv2.VideoCapture('CarCounter\Videos\cars.mp4')
   mask = cv2.imread("CarCounter\mask.png")
   template = cv2.imread("CarCounter\cover.png", cv2.IMREAD_UNCHANGED)
   ```
   The video feed and mask/template images are loaded and resized to the desired dimensions.

3. **Object Detection and Filtering**
   ```python
   results = model(imgMasked, stream=True)
   ```
   YOLO detects objects in the masked video frames. The script filters detections to focus on specific vehicle classes (car, truck, bus, motorbike) and confidence thresholds.

4. **Tracking Detected Objects**
   ```python
   resultsTracker = tracker.update(detections)
   ```
   SORT is used to track detected objects across frames, maintaining a unique ID for each tracked vehicle.

5. **Count Vehicles Passing a Line**
   ```python
   if limits[0] - 15 < cx < limits[0] + 15:
       if totalCount.count(id) == 0:
           totalCount.append(id)
   ```
   A virtual counting line is used to tally the number of vehicles passing through a defined region.

6. **Visualization**
   ```python
   cv2.imshow("image", img)
   ```
   The results, including bounding boxes, object labels, and counts, are displayed in real-time.

---

## Usage

### Running the Tracker
Run the main script with the following command:
```bash
python car_tracker.py --source <video_file_or_camera_id> --weights weights/yolov8n.pt
```

### Parameters
- `--source`: Path to the video file or camera ID (default: `0` for webcam).
- `--weights`: Path to the YOLOv8 Nano model weights file.
- `--save-output`: Save the output video (optional).

### Example
```bash
python car_tracker.py --source videos/sample.mp4 --weights weights/yolov8n.pt --save-output
```

---

## Repository Structure
```
car-tracking-yolo8nano/
├── car_tracker.py       # Main script for car tracking
├── requirements.txt     # Required Python packages
├── weights/             # Directory for YOLO model weights
├── videos/              # Directory for input videos
├── outputs/             # Directory for output videos
└── README.md            # Project documentation
```

---

## Model Details
**YOLOv8 Nano** is a highly optimized model for edge and low-resource environments. It offers excellent detection performance with minimal computational overhead, making it suitable for real-time applications on devices with limited hardware capabilities.

---

## Customization
- **Adjust Detection Thresholds**: Modify the confidence threshold to suit your application requirements.
- **Add Custom Classes**: Retrain the YOLO model to include specific objects beyond cars, trucks, and buses.
- **Integrate with IoT Devices**: Extend the project to send detection data to cloud services or IoT platforms.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your enhancements.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- [Ultralytics](https://github.com/ultralytics) for the YOLOv8 framework.
- OpenCV for video processing.
- SORT for robust object tracking.
