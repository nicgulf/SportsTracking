A computer vision application that tracks golf ball trajectories in videos using YOLO object detection and Kalman filtering, providing real-time analysis of distance, speed, and launch angle.

## Features

- **Real-time ball tracking** using YOLO v8 object detection
- **Kalman filter** for smooth trajectory prediction and noise reduction
- **Spline interpolation** for visually appealing trajectory visualization
- **Comprehensive metrics calculation**:
  - Distance traveled (meters and yards)
  - Average speed (km/h and m/s)
  - Launch angle estimation
- **Video overlay** with tracking status and live metrics
- **Robust detection** with configurable confidence thresholds

## Demo

The system processes golf videos and outputs annotated footage showing:
- Blue bounding box around detected golf ball
- Yellow prediction point from Kalman filter
- Red trajectory path with transparency overlay
- Real-time metrics display

## Requirements

```bash
pip install -q ultralytics opencv-python-headless scipy pytorch-lightning supervision
```

### Dependencies

- **ultralytics**: YOLO v8 model implementation
- **opencv-python**: Video processing and computer vision
- **numpy**: Numerical computations
- **scipy**: Spline interpolation for smooth trajectories
- **pytorch-lightning**: Simplifies model training and inference (optional if retraining)
- **supervision**: Used for overlay utilities and additional video annotations (if needed)

## Installation

1. Clone or download the project files
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained YOLO model file (`best_final.pt`) in the project directory
4. Update the input/output video paths in the main function

## Usage

### Basic Usage

```python
from golf_tracker import process_video_with_trajectory

# Process a golf video
process_video_with_trajectory("input_video.mp4", "output_video.mp4")
```

### Configuration

The system includes several configurable parameters:

```python
MODEL_PATH = "best_final.pt"           # Path to trained YOLO model
BALL_CLASS_ID = 1                      # Class ID for golf ball
CONFIDENCE_THRESHOLD = 0.38            # Detection confidence threshold
MIN_BOX_AREA = 20                      # Minimum bounding box area
MAX_TRAJECTORY_POINTS = 100            # Maximum trajectory history
KALMAN_MIN_FRAMES = 50                 # Frames before trajectory saving
MAX_POSITION_JUMP = 50                 # Max allowed position jump
PIXELS_PER_METER = 50                  # Calibration: pixels to meters
PIXELS_PER_YARD = 54.7                 # Calibration: pixels to yards
```

## How It Works

### 1. Object Detection
- Uses YOLO v8 to detect golf balls in each frame
- Filters detections by confidence threshold and minimum area
- Selects the best detection per frame based on confidence score

### 2. Kalman Filtering
- Initializes 4-state Kalman filter (x, y, dx, dy) on first detection
- Predicts ball position even when detection is lost
- Smooths trajectory by reducing noise and measurement errors

### 3. Trajectory Analysis
- Maintains history of ball positions using efficient deque structure
- Applies spline interpolation for smooth trajectory visualization
- Calculates metrics from launch point to landing point

### 4. Metrics Calculation

**Distance**: Euclidean distance from first to last trajectory point
```python
distance_px = np.sqrt(dx² + dy²)
distance_meters = distance_px / PIXELS_PER_METER
```

**Speed**: Average velocity based on Kalman filter predictions
```python
avg_speed = mean_velocity * fps / PIXELS_PER_METER * 3.6  # km/h
```

**Launch Angle**: Calculated from initial trajectory direction
```python
angle = arctan2(-delta_y, delta_x) * 180/π
```

## Model Training

The system requires a custom YOLO model trained on golf ball detection. Training details:

- **Class ID 1**: Golf ball class
- **Recommended dataset**: Golf ball images in various conditions
- **Model format**: PyTorch (.pt) file
- **Performance**: Optimized for small object detection

## Output Format

### Console Output
```
✅ Kalman initialized at frame 45
🎯 Distance: 187.50 m | 205.38 yd
🚀 Avg Speed: 28.75 km/h (7.99 m/s)
📀 Launch Angle: 12.3°
✅ Video processing complete.
```

### Video Overlay
- Frame counter and tracking status
- Real-time distance, speed, and angle measurements
- Visual trajectory with semi-transparent red line
- Detection bounding box and prediction point

## Calibration

For accurate measurements, calibrate the pixel-to-real-world conversion:

1. **Measure known distance** in your video (e.g., tee markers)
2. **Count pixels** for that distance
3. **Update constants**:
   ```python
   PIXELS_PER_METER = pixels_measured / meters_measured
   PIXELS_PER_YARD = pixels_measured / yards_measured
   ```

## Troubleshooting

### Common Issues

**No detections found**:
- Lower `CONFIDENCE_THRESHOLD`
- Check if `BALL_CLASS_ID` matches your model
- Verify model path and file integrity

**Erratic trajectory**:
- Increase `KALMAN_MIN_FRAMES` for more stability
- Adjust `MAX_POSITION_JUMP` for your video scale
- Check lighting conditions and ball visibility

**Inaccurate measurements**:
- Recalibrate `PIXELS_PER_METER` and `PIXELS_PER_YARD`
- Ensure consistent camera distance and angle
- Verify that the ball is clearly visible throughout flight

## Performance Optimization

- **GPU acceleration**: Enable CUDA for YOLO inference
- **Resolution**: Lower resolution videos process faster
- **Confidence threshold**: Higher values reduce false positives
- **Frame skipping**: Process every nth frame for real-time applications

## File Structure

```
golf-ball-tracker/
├── golf_tracker.py          # Main tracking script
├── best_final.pt           # Trained YOLO model
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── input_videos/          # Input video directory
└── output_videos/         # Processed video output
```

## Contributing

Contributions are welcome! Areas for improvement:
- Multi-ball tracking
- 3D trajectory estimation
- Real-time processing optimization
- Mobile app integration
- Advanced physics modeling

## License


## Acknowledgments

- **Ultralytics**: YOLO v8 implementation
- **OpenCV**: Computer vision framework
- **SciPy**: Mathematical computations

