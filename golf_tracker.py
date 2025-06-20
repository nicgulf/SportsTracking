"""
Golf Ball Trajectory Tracker

A computer vision system that tracks golf ball trajectories in videos using YOLO object detection
and Kalman filtering. Provides real-time analysis of distance, speed, and launch angle.

Author: [Your Name]
Date: [Date]
Version: 1.0
"""

# ============================================================================
# IMPORTS
# ============================================================================

from ultralytics import YOLO                    # YOLO object detection model
import cv2                                      # OpenCV for video processing
import numpy as np                              # Numerical operations
import scipy.interpolate                        # Spline interpolation for smooth trajectories
from collections import deque                   # Efficient double-ended queue
import math                                     # Mathematical utilities

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Model Configuration
MODEL_PATH = "best_final.pt"                    # Path to trained YOLO model file
BALL_CLASS_ID = 1                              # Class ID for golf ball in YOLO model

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.38                     # Minimum confidence score for valid detection
MIN_BOX_AREA = 20                              # Minimum bounding box area (filters noise)

# Tracking Parameters
MAX_TRAJECTORY_POINTS = 100                     # Maximum points stored in trajectory history
KALMAN_MIN_FRAMES = 50                         # Minimum frames before trajectory recording starts
MAX_POSITION_JUMP = 50                         # Maximum allowed position jump between frames (pixels)

# Calibration Constants
PIXELS_PER_METER = 50                          # Conversion factor: pixels to meters
PIXELS_PER_YARD = 54.7                         # Conversion factor: pixels to yards
INITIAL_DIRECTION_POINTS = 5                   # Number of points used for launch angle calculation

# ============================================================================
# KALMAN FILTER INITIALIZATION
# ============================================================================

def initialize_kalman_filter(cx, cy):
    """
    Initialize a 4-state Kalman filter for ball position tracking.
    
    The filter tracks: [x_position, y_position, x_velocity, y_velocity]
    
    Args:
        cx (float): Initial x-coordinate of the ball center
        cy (float): Initial y-coordinate of the ball center
    
    Returns:
        cv2.KalmanFilter: Configured Kalman filter instance
    """
    # Create Kalman filter: 4 states (x, y, dx, dy), 2 measurements (x, y)
    kalman = cv2.KalmanFilter(4, 2)
    
    # Measurement matrix: maps state vector to measurement vector
    # Only position (x, y) is directly measured, not velocity
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],  # x_measured = x_state
        [0, 1, 0, 0]   # y_measured = y_state
    ], np.float32)
    
    # State transition matrix: defines how state evolves over time
    # Assumes constant velocity model: new_pos = old_pos + velocity
    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],  # x_new = x_old + dx
        [0, 1, 0, 1],  # y_new = y_old + dy
        [0, 0, 1, 0],  # dx_new = dx_old (constant velocity)
        [0, 0, 0, 1]   # dy_new = dy_old (constant velocity)
    ], np.float32)
    
    # Process noise covariance: uncertainty in state prediction
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    
    # Measurement noise covariance: uncertainty in measurements
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32)
    
    # Initialize state vectors with starting position and zero velocity
    initial_state = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    kalman.statePre = initial_state.copy()          # Predicted state
    kalman.statePost = initial_state.copy()         # Corrected state
    
    return kalman

# ============================================================================
# TRAJECTORY VISUALIZATION
# ============================================================================

def draw_smoothed_trajectory(frame, trajectory):
    """
    Draw a smooth spline-interpolated trajectory on the video frame.
    
    Uses scipy spline interpolation to create a visually appealing trajectory
    line with transparency overlay.
    
    Args:
        frame (numpy.ndarray): Video frame to draw on
        trajectory (deque): Collection of (x, y) trajectory points
    """
    # Need minimum points for spline interpolation
    if len(trajectory) < 6:
        return
    
    # Create overlay for transparency effect
    overlay = frame.copy()
    points = np.array(trajectory)
    
    # Create time parameter for spline interpolation
    t = np.arange(len(points))
    
    # Generate separate splines for x and y coordinates
    # s=2 controls smoothing factor (higher = smoother but less accurate)
    spline_x = scipy.interpolate.UnivariateSpline(t, points[:, 0], s=2)
    spline_y = scipy.interpolate.UnivariateSpline(t, points[:, 1], s=2)
    
    # Create dense set of interpolated points for smooth curve
    t_smooth = np.linspace(t.min(), t.max(), len(points) * 3)
    smooth_pts = np.stack((spline_x(t_smooth), spline_y(t_smooth)), axis=1).astype(int)
    
    # Draw thick red trajectory line on overlay
    for i in range(1, len(smooth_pts)):
        pt1 = tuple(smooth_pts[i - 1])
        pt2 = tuple(smooth_pts[i])
        cv2.line(overlay, pt1, pt2, (0, 0, 255), 8)  # BGR: Red color, thickness 8
    
    # Blend overlay with original frame for transparency effect
    alpha = 0.5  # Transparency factor (0.0 = transparent, 1.0 = opaque)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# ============================================================================
# MAIN VIDEO PROCESSING FUNCTION
# ============================================================================

def process_video_with_trajectory(input_path, output_path):
    """
    Main function to process golf video and track ball trajectory.
    
    Performs the complete pipeline:
    1. Load YOLO model and open video
    2. Detect golf ball in each frame
    3. Track ball using Kalman filter
    4. Calculate trajectory metrics
    5. Overlay results on video
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path for output video with overlays
    """
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    # Load pre-trained YOLO model
    model = YOLO(MODEL_PATH)
    
    # Open input video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {input_path}")
        return
    
    # Extract video properties for output video writer
    fps = cap.get(cv2.CAP_PROP_FPS)                 # Frames per second
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # ========================================================================
    # TRACKING STATE VARIABLES
    # ========================================================================
    
    frame_index = 0                                 # Current frame counter
    kalman = None                                   # Kalman filter instance
    prev_prediction = None                          # Previous frame prediction
    kalman_initialized = False                      # Kalman filter status
    tracking_active = False                         # Overall tracking status
    stable_frame_count = 0                          # Frames since last detection
    first_detection_occurred = False                # Flag for first ball detection
    
    # Data structures for tracking history
    recent_positions = deque(maxlen=5)              # Recent detection positions
    trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS) # Full trajectory history
    velocities = []                                 # Velocity measurements
    
    print(f"🎬 Processing video: {input_path}")
    print(f"📊 Video properties: {frame_width}x{frame_height} @ {fps:.1f} FPS ({total_frames} frames)")
    
    # ========================================================================
    # MAIN PROCESSING LOOP
    # ========================================================================
    
    while True:
        # Read next frame from video
        ret, frame = cap.read()
        if not ret:
            break  # End of video reached
        
        frame_index += 1
        
        # ====================================================================
        # BALL DETECTION USING YOLO
        # ====================================================================
        
        # Run YOLO inference on current frame
        # Filter for ball class only to improve performance
        results = model(frame, classes=[BALL_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        best_ball = None  # Will store the best detection this frame
        
        # Process all detections in current frame
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Verify this is a ball detection
                    if int(box.cls[0]) == BALL_CLASS_ID:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Filter out very small detections (likely noise)
                        area = (x2 - x1) * (y2 - y1)
                        if area >= MIN_BOX_AREA:
                            # Calculate center point of bounding box
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            confidence = float(box.conf[0])
                            
                            # Keep detection with highest confidence
                            if best_ball is None or confidence > best_ball[2]:
                                best_ball = (cx, cy, confidence, x1, y1, x2, y2)
        
        # ====================================================================
        # KALMAN FILTER PROCESSING
        # ====================================================================
        
        # Process valid ball detection
        if best_ball:
            cx, cy, confidence, x1, y1, x2, y2 = best_ball
            
            # Draw detection bounding box (blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Initialize Kalman filter on first detection
            if not kalman_initialized:
                kalman = initialize_kalman_filter(cx, cy)
                kalman_initialized = True
                stable_frame_count = 0
                tracking_active = True
                first_detection_occurred = True
                print(f"✅ Kalman filter initialized at frame {frame_index}")
            
            # Update Kalman filter with current measurement
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            kalman.correct(measurement)
            
            # Store recent positions for stability checking
            recent_positions.append((cx, cy))
            stable_frame_count += 1
        
        # Generate prediction even if no detection (tracking through occlusion)
        if kalman_initialized:
            # Predict next position using Kalman filter
            prediction = kalman.predict()
            px, py = int(prediction[0]), int(prediction[1])
            
            # Calculate velocity from prediction changes
            if prev_prediction is not None:
                dx = prediction[0] - prev_prediction[0]
                dy = prediction[1] - prev_prediction[1]
                velocities.append((dx, dy))
            prev_prediction = prediction.copy()
            
            # Add prediction to trajectory if conditions are met
            if stable_frame_count >= KALMAN_MIN_FRAMES:
                # Include point if we have detection OR prediction is close to recent positions
                should_include = False
                
                if best_ball:
                    should_include = True  # Always include if we have detection
                elif recent_positions:
                    # Include if predicted position is reasonable
                    last_pos = np.array(recent_positions[-1])
                    pred_pos = np.array([px, py])
                    distance = np.linalg.norm(pred_pos - last_pos)
                    should_include = distance < MAX_POSITION_JUMP
                
                if should_include:
                    trajectory.append((px, py))
            
            # Draw prediction point (yellow)
            cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
        
        # ====================================================================
        # TRAJECTORY VISUALIZATION
        # ====================================================================
        
        # Draw smooth trajectory line
        draw_smoothed_trajectory(frame, trajectory)
        
        # ====================================================================
        # METRICS CALCULATION AND DISPLAY
        # ====================================================================
        
        if first_detection_occurred and len(trajectory) >= 2:
            # Calculate trajectory endpoints
            launch_point = np.array(trajectory[0])      # First trajectory point
            landing_point = np.array(trajectory[-1])    # Last trajectory point
            
            # Calculate total distance traveled
            dx = landing_point[0] - launch_point[0]
            dy = landing_point[1] - launch_point[1]
            distance_pixels = np.sqrt(dx**2 + dy**2)
            
            # Convert to real-world units
            distance_meters = distance_pixels / PIXELS_PER_METER
            distance_yards = distance_pixels / PIXELS_PER_YARD
            
            # Calculate launch angle
            launch_angle = 0
            if len(trajectory) >= INITIAL_DIRECTION_POINTS:
                # Use initial trajectory segment for angle calculation
                direction_point = np.array(trajectory[INITIAL_DIRECTION_POINTS - 1])
                delta = direction_point - launch_point
                
                # Calculate angle (note: y-axis is flipped in image coordinates)
                launch_angle = np.degrees(np.arctan2(-delta[1], delta[0]))
                
                # Normalize to 0-360 degree range
                if launch_angle < 0:
                    launch_angle += 360
            
            # Calculate average speed
            ball_speed_kph = ball_speed_mps = 0
            if velocities:
                # Average velocity components
                avg_dx = np.mean([v[0] for v in velocities])
                avg_dy = np.mean([v[1] for v in velocities])
                
                # Convert to pixels per second
                pixels_per_second = np.linalg.norm([avg_dx, avg_dy]) * fps
                
                # Convert to real-world speed units
                ball_speed_mps = pixels_per_second / PIXELS_PER_METER  # meters per second
                ball_speed_kph = ball_speed_mps * 3.6                  # kilometers per hour
            
            # ================================================================
            # ON-SCREEN DISPLAY
            # ================================================================
            
            # Display metrics on video frame
            text_color = (255, 255, 0)  # Yellow text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Distance information
            distance_text = f"Distance: {distance_meters:.2f} m / {distance_yards:.2f} yd"
            cv2.putText(frame, distance_text, (10, 60), font, font_scale, text_color, thickness)
            
            # Speed information
            speed_text = f"Speed: {ball_speed_kph:.2f} km/h ({ball_speed_mps:.2f} m/s)"
            cv2.putText(frame, speed_text, (10, 90), font, font_scale, text_color, thickness)
            
            # Launch angle information
            angle_text = f"Launch Angle: {launch_angle:.1f}°"
            cv2.putText(frame, angle_text, (10, 120), font, font_scale, text_color, thickness)
            
            # ================================================================
            # FINAL RESULTS LOGGING
            # ================================================================
            
            # Print final results when processing is complete
            if frame_index == total_frames:
                print(f"\n🎯 Final Results:")
                print(f"   Distance: {distance_meters:.2f} m | {distance_yards:.2f} yd")
                print(f"🚀 Average Speed: {ball_speed_kph:.2f} km/h ({ball_speed_mps:.2f} m/s)")
                print(f"📐 Launch Angle: {launch_angle:.1f}°")
        
        # ====================================================================
        # STATUS DISPLAY
        # ====================================================================
        
        # Show current tracking status
        status = "Tracking" if tracking_active else "Searching"
        status_text = f"Frame: {frame_index}/{total_frames} | Status: {status}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress indicator
        if frame_index % 100 == 0:
            progress = (frame_index / total_frames) * 100
            print(f"⏳ Processing: {progress:.1f}% ({frame_index}/{total_frames} frames)")
        
        # Write processed frame to output video
        out.write(frame)
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    # Release video resources
    cap.release()
    out.release()
    
    print(f"✅ Video processing complete!")
    print(f"📁 Output saved to: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for the golf ball tracker.
    
    Modify the input and output paths below to process your golf videos.
    """
    # Configuration for video processing
    INPUT_VIDEO = "59.mp4"                          # Input golf video file
    OUTPUT_VIDEO = "output_59.mp4"                  # Output video with tracking overlay
    
    print("🏌️‍♂️ Golf Ball Trajectory Tracker")
    print("=" * 50)
    
    # Process the video
    process_video_with_trajectory(INPUT_VIDEO, OUTPUT_VIDEO)
    
    print("=" * 50)