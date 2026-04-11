import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.spatial.distance import euclidean
import math

# MediaPipe imports - support both legacy and new API
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import BaseOptions
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_LEGACY = False  # We use the new Tasks API
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_LEGACY = False

# Use OpenCV DNN-based pose estimation only if MediaPipe not available
USE_OPENCV_POSE = not MEDIAPIPE_AVAILABLE

# Fallback landmark layout: fractions of bounding box height (bh) or width (bw).
# All offsets are scale-invariant (fraction of bh/bw), not pixel values.
BBOX_HEAD_CENTER_Y_FRAC = 0.15   # Head center vertical position (from bbox top)
BBOX_EYE_HORIZONTAL_FRAC = 0.02  # Eye horizontal offset from nose (fraction of bh)
BBOX_EYE_ABOVE_NOSE_FRAC = 0.02  # Eye vertical offset above nose
BBOX_EAR_HORIZONTAL_FRAC = 0.05  # Ear horizontal offset from nose
BBOX_EAR_VERTICAL_FRAC = 0.01    # Ear vertical offset from nose
BBOX_MOUTH_BELOW_NOSE_FRAC = 0.04   # Mouth center below nose
BBOX_MOUTH_HALF_WIDTH_FRAC = 0.04   # Mouth half-width
BBOX_SHOULDER_Y_FRAC = 0.25
BBOX_SHOULDER_HALF_WIDTH_FRAC = 0.4  # half-width from center, as fraction of bw
BBOX_ELBOW_Y_FRAC = 0.45
BBOX_ELBOW_WIDTH_FRAC = 1.2   # multiplier of shoulder half-width
BBOX_WRIST_Y_FRAC = 0.60
BBOX_WRIST_WIDTH_FRAC = 1.3
BBOX_HIP_Y_FRAC = 0.50
BBOX_HIP_HALF_WIDTH_FRAC = 0.2  # half-width from center, as fraction of bw
BBOX_HIP_DROP_FRAC = 0.05     # Vertical drop from hip pointer to hip joint
BBOX_KNEE_Y_FRAC = 0.70
BBOX_KNEE_WIDTH_FRAC = 0.9
BBOX_ANKLE_Y_FRAC = 0.90
BBOX_ANKLE_WIDTH_FRAC = 0.8
BBOX_FOOT_EXTEND_FRAC = 0.02  # Heel/toe extend below ankle
BBOX_TOE_FORWARD_FRAC = 0.04


@dataclass
class EnhancedGestureFeature:
    """Enhanced representation of features extracted from a tennis gesture"""
    pose_landmarks: np.ndarray  # MediaPipe pose landmarks
    optical_flow: np.ndarray   # Motion vectors from optical flow
    motion_history: np.ndarray # Motion history image
    hog_features: np.ndarray   # Histogram of Oriented Gradients
    joint_angles: List[float]  # Computed joint angles
    trajectories: List[List[Tuple[float, float]]]  # Movement paths
    velocity_vectors: List[np.ndarray]  # Movement vectors between frames
    acceleration: List[float]  # Acceleration values
    temporal_keypoints: List[Dict]  # Key temporal moments in gesture


class EnhancedTennisGestureAnalyzer:
    def __init__(self, use_opencv_pose: bool = False):
        """
        Initialize the Tennis Gesture Analyzer.

        Args:
            use_opencv_pose: If True, use OpenCV DNN-based pose estimation.
                           If False, use MediaPipe (default).
        """
        self.use_opencv_pose = use_opencv_pose or USE_OPENCV_POSE
        self.pose_detector = None

        if self.use_opencv_pose:
            # Initialize OpenCV DNN-based pose estimator
            print("Using OpenCV-based pose estimation (placeholder)")
        elif MEDIAPIPE_AVAILABLE:
            # Initialize MediaPipe Tasks API Pose Landmarker
            try:
                base_options = BaseOptions(
                    model_asset_path=''  # Uses default model
                )
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.pose_detector = vision.PoseLandmarker.create_from_options(options)
                print("MediaPipe Pose Landmarker initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize MediaPipe pose detector: {e}")
                self.pose_detector = None
        else:
            print("Warning: MediaPipe not available. Using fallback mode.")

        # Store previous frame landmarks and velocity for interpolation when detection fails
        self.prev_landmarks: Optional[np.ndarray] = None
        self.prev_landmarks_confidence: float = 0.0
        self.landmark_velocity: Optional[np.ndarray] = None  # (33, 2) per-frame displacement

        # Initialize optical flow for motion features
        self.opt_flow = cv2.calcOpticalFlowFarneback

        # Initialize HOG descriptor for motion features
        self.hog = cv2.HOGDescriptor()

        # Initialize the gesture database
        self.gesture_database = {}

    def calculate_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Calculate optical flow between two consecutive frames"""
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        return flow

    def calculate_motion_history(self, frame_sequence: List[np.ndarray], current_idx: int) -> np.ndarray:
        """Calculate motion history image from recent frames"""
        if current_idx < 2:
            # Return zero array if not enough frames
            h, w = frame_sequence[0].shape[:2]
            return np.zeros((h, w), dtype=np.float32)

        # Take a window of frames around current index
        start_idx = max(0, current_idx - 5)
        end_idx = min(current_idx + 1, len(frame_sequence))

        motion_history = np.zeros_like(frame_sequence[0][:, :, 0], dtype=np.float32)

        for i in range(start_idx + 1, end_idx):
            prev_frame = cv2.cvtColor(frame_sequence[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frame_sequence[i], cv2.COLOR_BGR2GRAY)

            # Compute difference to identify motion regions
            diff = cv2.absdiff(prev_frame.astype(np.float32), curr_frame.astype(np.float32))
            diff = np.where(diff > 30, diff, 0)  # Threshold to reduce noise

            # Add to motion history with exponential decay
            motion_history = 0.7 * motion_history + 0.3 * diff

        return motion_history

    def extract_hog_features(self, frame: np.ndarray, bbox=None) -> np.ndarray:
        """Extract HOG features from frame or region of interest"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If no bounding box provided, use the entire frame
        if bbox is None:
            roi = gray_frame
        else:
            x, y, w, h = bbox
            roi = gray_frame[y:y+h, x:x+w]

        # Resize to standard size for HOG
        roi_resized = cv2.resize(roi, (128, 128))

        # Extract HOG features
        features = self.hog.compute(roi_resized)

        return features.flatten() if features is not None else np.array([])

    def calculate_joint_angles(self, landmarks: np.ndarray) -> List[float]:
        """
        Calculate angles between key joints using MediaPipe Pose landmarks.

        MediaPipe provides 33 landmarks:
        - 0-10: Face/head
        - 11-22: Upper body (shoulders, elbows, wrists)
        - 23-32: Lower body (hips, knees, ankles)

        Returns angles in degrees for tennis-specific joint analysis.
        """
        if len(landmarks) < 33:  # Not enough landmarks
            return [0.0] * 12  # Return default angles

        angles = []

        # Define triplets of joints to calculate angles for (MediaPipe indices)
        angle_triplets = [
            # Right arm angles (dominant arm for most players)
            (12, 14, 16),  # Right shoulder-elbow-wrist (elbow flexion)
            (11, 12, 14),  # Left shoulder-right shoulder-elbow (shoulder abduction)

            # Left arm angles
            (11, 13, 15),  # Left shoulder-elbow-wrist (elbow flexion)
            (12, 11, 13),  # Right shoulder-left shoulder-elbow (shoulder abduction)

            # Right leg angles (stance analysis)
            (24, 26, 28),  # Right hip-knee-ankle (knee flexion)
            (23, 24, 26),  # Left hip-right hip-knee (hip angle)

            # Left leg angles
            (23, 25, 27),  # Left hip-knee-ankle (knee flexion)
            (24, 23, 25),  # Right hip-left hip-knee (hip angle)

            # Torso/shoulder rotation (critical for tennis strokes)
            (11, 12, 24),  # Left shoulder-right shoulder-right hip
            (12, 11, 23),  # Right shoulder-left shoulder-left hip

            # Body lean during stroke
            (12, 11, 23),  # Right shoulder-left shoulder-left hip
            (11, 12, 24),  # Left shoulder-right shoulder-right hip
        ]

        for triplet in angle_triplets:
            p1_idx, p2_idx, p3_idx = triplet
            try:
                if p1_idx < len(landmarks) and p2_idx < len(landmarks) and p3_idx < len(landmarks):
                    # Extract landmark coordinates
                    point1 = np.array([landmarks[p1_idx][0], landmarks[p1_idx][1]])
                    point2 = np.array([landmarks[p2_idx][0], landmarks[p2_idx][1]])
                    point3 = np.array([landmarks[p3_idx][0], landmarks[p3_idx][1]])

                    # Calculate vectors
                    v1 = point1 - point2
                    v2 = point3 - point2

                    # Calculate angle in radians, then convert to degrees
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    angle_deg = np.degrees(angle_rad)
                    angles.append(angle_deg)
                else:
                    angles.append(0.0)  # Default if indices are out of bounds
            except Exception:
                angles.append(0.0)  # Default if calculation fails

        return angles

    def extract_trajectories(self, frame_sequence: List[np.ndarray], current_idx: int,
                            landmarks: np.ndarray, cached_landmarks: Optional[Dict[int, np.ndarray]] = None) -> List[List[Tuple[float, float]]]:
        """
        Extract movement trajectories of key joints over recent frames.

        Args:
            frame_sequence: List of video frames
            current_idx: Current frame index
            landmarks: Landmarks for current frame (fallback)
            cached_landmarks: Optional dict of pre-computed landmarks per frame index

        Returns:
            List of trajectories for each key joint (wrist, elbow, shoulder positions)
        """
        trajectory_length = min(15, current_idx + 1)  # Track last 15 frames
        trajectories = []

        # Define key joints to track (MediaPipe indices)
        key_joint_indices = [
            15,  # Left wrist
            16,  # Right wrist (critical for racquet tracking)
            13,  # Left elbow
            14,  # Right elbow
            11,  # Left shoulder
            12,  # Right shoulder
        ]

        for joint_idx in key_joint_indices:
            if joint_idx >= len(landmarks):
                continue

            trajectory = []
            start_frame_idx = max(0, current_idx - trajectory_length)

            for frame_idx in range(start_frame_idx, current_idx + 1):
                # Try to use cached landmarks first
                frame_landmarks = None
                if cached_landmarks is not None and frame_idx in cached_landmarks:
                    frame_landmarks = cached_landmarks[frame_idx]

                # Fall back to extracting from frame if not cached
                if frame_landmarks is None:
                    frame_landmarks = self.extract_landmarks_from_frame(frame_sequence[frame_idx])
                    # Cache for next time
                    if cached_landmarks is not None:
                        cached_landmarks[frame_idx] = frame_landmarks

                if frame_landmarks is not None and joint_idx < len(frame_landmarks):
                    x, y = frame_landmarks[joint_idx][0], frame_landmarks[joint_idx][1]
                    trajectory.append((x, y))
                elif trajectory:
                    # If we have previous points, extend with last known position
                    trajectory.append(trajectory[-1])

            trajectories.append(trajectory)

        return trajectories

    def extract_landmarks_from_frame(self, frame: np.ndarray,
                                     person_bbox: Optional[Tuple[int, int, int, int, float]] = None
                                     ) -> Optional[np.ndarray]:
        """
        Extract 33 pose landmarks from a frame.

        Uses MediaPipe Pose when available, otherwise falls back to OpenCV-based
        detection using the provided person bounding box.

        Args:
            frame: Input frame (BGR)
            person_bbox: Optional (x, y, w, h, confidence) tuple from person detector

        Returns:
            Normalized coordinates (x, y) in range [0, 1] for valid detections,
            or None if no valid pose is detected.
        """
        if not self.use_opencv_pose and MEDIAPIPE_LEGACY:
            return self._extract_landmarks_medipipe(frame)
        else:
            return self._extract_landmarks_fallback(frame, person_bbox)

    def _extract_landmarks_medipipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 33 MediaPipe pose landmarks from a frame.

        Returns normalized coordinates (x, y) in range [0, 1] for valid detections,
        or None if no valid pose is detected.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks is None:
            # No pose detected - velocity-based interpolation from previous landmarks
            if self.prev_landmarks is not None and self.prev_landmarks_confidence > 0.3:
                self.prev_landmarks_confidence *= 0.9  # Decay confidence
                if self.landmark_velocity is not None:
                    interpolated = self.prev_landmarks + self.landmark_velocity
                    interpolated = np.clip(interpolated, 0.0, 1.0)
                else:
                    interpolated = self.prev_landmarks.copy()
                self.prev_landmarks = interpolated
                if self.landmark_velocity is not None:
                    self.landmark_velocity *= 0.95  # Decay velocity so we don't extrapolate forever
                return interpolated
            return None

        # Extract landmarks from results
        h, w = frame.shape[:2]
        landmarks = []

        for landmark in results.pose_landmarks.landmark:
            # MediaPipe already provides normalized coordinates
            x = np.clip(landmark.x, 0, 1)
            y = np.clip(landmark.y, 0, 1)
            visibility = landmark.visibility  # Confidence score (0-1)

            landmarks.append([x, y])

        landmarks_array = np.array(landmarks, dtype=np.float32)

        # Update velocity from displacement (before overwriting prev_landmarks)
        if self.prev_landmarks is not None and self.prev_landmarks.shape == landmarks_array.shape:
            self.landmark_velocity = landmarks_array - self.prev_landmarks
        self.prev_landmarks = landmarks_array.copy()
        self.prev_landmarks_confidence = 1.0

        return landmarks_array

    def _extract_landmarks_fallback(self, frame: np.ndarray,
                                    person_bbox: Optional[Tuple[int, int, int, int, float]] = None
                                    ) -> Optional[np.ndarray]:
        """
        Fallback landmark extraction when MediaPipe is not available.

        Uses the provided person bounding box for accurate landmark placement.

        Args:
            frame: Input frame (BGR)
            person_bbox: Optional (x, y, w, h, confidence) tuple from person detector

        Returns:
            33x2 array of normalized landmarks or None
        """
        h, w = frame.shape[:2]

        # Use provided person bounding box if available
        if person_bbox is not None:
            x, y, bw, bh, conf = person_bbox

            # Check if bbox position changed significantly from previous
            if self.prev_landmarks is not None and len(self.prev_landmarks) > 0:
                # Calculate previous center from landmarks
                prev_center_x = np.mean(self.prev_landmarks[:, 0]) * w
                prev_center_y = np.mean(self.prev_landmarks[:, 1]) * h
                new_center_x = (x + bw / 2)
                new_center_y = (y + bh / 2)

                dist = np.sqrt((new_center_x - prev_center_x)**2 + (new_center_y - prev_center_y)**2)

                # If bbox moved more than 200 pixels, reset landmarks (new person detected)
                if dist > 200:
                    self.prev_landmarks = None
                    self.prev_landmarks_confidence = 0

            # Create approximate 33 landmarks based on bounding box
            landmarks = self._create_landmarks_from_bbox(x, y, bw, bh, w, h)
            self.prev_landmarks = landmarks.copy()
            self.prev_landmarks_confidence = conf * 0.8  # Scale confidence
            return landmarks

        # No bbox provided - use temporal interpolation from previous landmarks
        # This is better than detecting the wrong person!
        if self.prev_landmarks is not None and self.prev_landmarks_confidence > 0.4:
            self.prev_landmarks_confidence *= 0.92  # Decay confidence

            # Return interpolated landmarks with small jitter for natural movement
            if self.prev_landmarks_confidence >= 0.3:
                return self.prev_landmarks * 0.985 + np.random.normal(0, 0.003, self.prev_landmarks.shape)

        # Confidence too low - clear and return None
        # Don't try to detect from contours - that picks up wrong person
        self.prev_landmarks = None
        self.prev_landmarks_confidence = 0
        return None

    def _create_landmarks_from_bbox(self, x: int, y: int, bw: int, bh: int,
                                   img_w: int, img_h: int) -> np.ndarray:
        """Create approximate 33 landmarks from a bounding box.

        All positions use fractions of bh/bw so the layout scales with person size.
        """
        landmarks = np.zeros((33, 2), dtype=np.float32)
        cx = x + bw / 2

        # Scale-invariant offsets (fractions of body height)
        eye_off_x = bh * BBOX_EYE_HORIZONTAL_FRAC
        eye_off_y = bh * BBOX_EYE_ABOVE_NOSE_FRAC
        ear_off_x = bh * BBOX_EAR_HORIZONTAL_FRAC
        ear_off_y = bh * BBOX_EAR_VERTICAL_FRAC
        mouth_off_y = bh * BBOX_MOUTH_BELOW_NOSE_FRAC
        mouth_half_w = bh * BBOX_MOUTH_HALF_WIDTH_FRAC

        # Head (landmarks 0-10)
        head_y = y + bh * BBOX_HEAD_CENTER_Y_FRAC
        head_x = cx
        landmarks[0] = [head_x / img_w, head_y / img_h]  # Nose
        landmarks[1] = [(head_x - eye_off_x) / img_w, (head_y - eye_off_y) / img_h]  # Left eye
        landmarks[2] = [(head_x + eye_off_x) / img_w, (head_y - eye_off_y) / img_h]  # Right eye
        landmarks[3] = [(head_x - ear_off_x) / img_w, (head_y - ear_off_y) / img_h]  # Left ear
        landmarks[4] = [(head_x + ear_off_x) / img_w, (head_y - ear_off_y) / img_h]  # Right ear
        mouth_y = head_y + mouth_off_y
        for i in range(5, 11):  # Mouth (6 points)
            t = (i - 5) / 5.0  # 0 to 1
            mouth_x = head_x - mouth_half_w + (2 * mouth_half_w * t)
            landmarks[i] = [mouth_x / img_w, mouth_y / img_h]

        # Upper body (11-20)
        shoulder_y = y + bh * BBOX_SHOULDER_Y_FRAC
        shoulder_w = bw * BBOX_SHOULDER_HALF_WIDTH_FRAC
        landmarks[11] = [(cx - shoulder_w) / img_w, shoulder_y / img_h]  # Left shoulder
        landmarks[12] = [(cx + shoulder_w) / img_w, shoulder_y / img_h]  # Right shoulder

        elbow_y = y + bh * BBOX_ELBOW_Y_FRAC
        elbow_w = shoulder_w * BBOX_ELBOW_WIDTH_FRAC
        landmarks[13] = [(cx - elbow_w) / img_w, elbow_y / img_h]  # Left elbow
        landmarks[14] = [(cx + elbow_w) / img_w, elbow_y / img_h]  # Right elbow

        wrist_y = y + bh * BBOX_WRIST_Y_FRAC
        wrist_w = shoulder_w * BBOX_WRIST_WIDTH_FRAC
        landmarks[15] = [(cx - wrist_w) / img_w, wrist_y / img_h]  # Left wrist
        landmarks[16] = [(cx + wrist_w) / img_w, wrist_y / img_h]  # Right wrist

        for i in range(17, 21):
            landmarks[i] = landmarks[15 if i < 19 else 16]

        # Lower body (21-32)
        hip_y = y + bh * BBOX_HIP_Y_FRAC
        hip_w = bw * BBOX_HIP_HALF_WIDTH_FRAC
        hip_drop = bh * BBOX_HIP_DROP_FRAC
        landmarks[21] = [(cx - hip_w) / img_w, hip_y / img_h]  # Left hip pointer
        landmarks[22] = [(cx + hip_w) / img_w, hip_y / img_h]  # Right hip pointer
        landmarks[23] = [(cx - hip_w) / img_w, (hip_y + hip_drop) / img_h]  # Left hip
        landmarks[24] = [(cx + hip_w) / img_w, (hip_y + hip_drop) / img_h]  # Right hip

        knee_y = y + bh * BBOX_KNEE_Y_FRAC
        knee_w = hip_w * BBOX_KNEE_WIDTH_FRAC
        landmarks[25] = [(cx - knee_w) / img_w, knee_y / img_h]  # Left knee
        landmarks[26] = [(cx + knee_w) / img_w, knee_y / img_h]  # Right knee

        ankle_y = y + bh * BBOX_ANKLE_Y_FRAC
        ankle_w = hip_w * BBOX_ANKLE_WIDTH_FRAC
        landmarks[27] = [(cx - ankle_w) / img_w, ankle_y / img_h]  # Left ankle
        landmarks[28] = [(cx + ankle_w) / img_w, ankle_y / img_h]  # Right ankle

        foot_extend = bh * BBOX_FOOT_EXTEND_FRAC
        toe_forward = bh * BBOX_TOE_FORWARD_FRAC
        landmarks[29] = [(cx - ankle_w * 1.1) / img_w, (ankle_y + foot_extend) / img_h]  # Left heel
        landmarks[30] = [(cx - ankle_w * 0.9) / img_w, (ankle_y + toe_forward) / img_h]  # Left toe
        landmarks[31] = [(cx + ankle_w * 0.9) / img_w, (ankle_y + toe_forward) / img_h]  # Right toe
        landmarks[32] = [(cx + ankle_w * 1.1) / img_w, (ankle_y + foot_extend) / img_h]  # Right heel

        return landmarks

    def draw_landmarks(self, frame: np.ndarray, landmarks: Optional[np.ndarray]) -> np.ndarray:
        """
        Visualize landmarks on the frame for debugging.
        """
        if landmarks is None:
            return frame

        output = frame.copy()
        h, w = frame.shape[:2]

        # Define colors for different body parts
        colors = {
            'head': (0, 0, 255),      # Red
            'torso': (0, 255, 0),     # Green
            'arm': (255, 0, 0),       # Blue
            'leg': (255, 255, 0)      # Cyan
        }

        # MediaPipe landmark indices
        NOSE = 0
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_ELBOW, RIGHT_ELBOW = 13, 14
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28

        # Draw circles at landmark positions
        for i, (x, y) in enumerate(landmarks):
            px, py = int(x * w), int(y * h)

            # Color based on body region
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Head/face
                color = colors['head']
            elif i in [11, 12, 21, 22, 23, 24]:  # Torso
                color = colors['torso']
            elif i in [13, 14, 15, 16, 17, 18, 19, 20]:  # Arms
                color = colors['arm']
            elif i in [25, 26, 27, 28, 29, 30, 31, 32]:  # Legs
                color = colors['leg']
            else:
                color = (255, 255, 255)  # White for others

            cv2.circle(output, (px, py), 5, color, -1)

        # Draw skeleton connections
        connections = [
            (LEFT_SHOULDER, LEFT_ELBOW, colors['arm']),
            (LEFT_ELBOW, LEFT_WRIST, colors['arm']),
            (RIGHT_SHOULDER, RIGHT_ELBOW, colors['arm']),
            (RIGHT_ELBOW, RIGHT_WRIST, colors['arm']),
            (LEFT_SHOULDER, LEFT_HIP, colors['torso']),
            (RIGHT_SHOULDER, RIGHT_HIP, colors['torso']),
            (LEFT_HIP, LEFT_KNEE, colors['leg']),
            (LEFT_KNEE, LEFT_ANKLE, colors['leg']),
            (RIGHT_HIP, RIGHT_KNEE, colors['leg']),
            (RIGHT_KNEE, RIGHT_ANKLE, colors['leg']),
            (LEFT_HIP, RIGHT_HIP, colors['torso']),
            (LEFT_SHOULDER, RIGHT_SHOULDER, colors['torso']),
        ]

        for idx1, idx2, color in connections:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                pt1 = (int(landmarks[idx1][0] * w), int(landmarks[idx1][1] * h))
                pt2 = (int(landmarks[idx2][0] * w), int(landmarks[idx2][1] * h))
                cv2.line(output, pt1, pt2, color, 2)

        return output

    def calculate_velocities_and_acceleration(self, landmarks: np.ndarray, prev_landmarks: np.ndarray,
                                           prev_velocity: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """Calculate velocity and acceleration from landmark positions"""
        if prev_landmarks is not None:
            # Calculate velocity as change in position
            velocity = landmarks - prev_landmarks

            # Calculate acceleration as change in velocity
            if prev_velocity is not None:
                acceleration = np.mean(np.abs(velocity - prev_velocity))
            else:
                acceleration = 0.0

            return velocity, acceleration

        return np.zeros_like(landmarks), 0.0

    def extract_enhanced_gesture_features(self, frame_sequence: List[np.ndarray]) -> List[EnhancedGestureFeature]:
        """
        Extract enhanced features from a sequence of video frames.

        This method processes all frames to extract:
        - 33 MediaPipe pose landmarks (normalized coordinates)
        - Optical flow between consecutive frames
        - Motion history images
        - HOG features for shape analysis
        - Joint angles for biomechanical analysis
        - Trajectories of key joints (wrists, elbows, shoulders)
        - Velocity and acceleration vectors

        Returns:
            List of EnhancedGestureFeature objects for each frame with valid pose detection
        """
        features = []
        prev_landmarks = None
        prev_velocity = None

        # First pass: extract and cache all landmarks
        cached_landmarks: Dict[int, np.ndarray] = {}
        for i, frame in enumerate(frame_sequence):
            landmarks = self.extract_landmarks_from_frame(frame)
            if landmarks is not None and len(landmarks) > 0:
                cached_landmarks[i] = landmarks

        # Second pass: extract all features using cached landmarks
        for i, frame in enumerate(frame_sequence):
            # Skip frames without valid landmarks
            if i not in cached_landmarks:
                continue

            landmarks = cached_landmarks[i]

            # Calculate optical flow (needs previous frame)
            optical_flow = np.array([])
            if i > 0:
                optical_flow = self.calculate_optical_flow(frame_sequence[i-1], frame)

            # Calculate motion history image
            motion_history = self.calculate_motion_history(frame_sequence, i)

            # Extract HOG features
            hog_features = self.extract_hog_features(frame)

            # Calculate joint angles
            joint_angles = self.calculate_joint_angles(landmarks)

            # Extract trajectories using cached landmarks
            trajectories = self.extract_trajectories(
                frame_sequence, i, landmarks, cached_landmarks
            )

            # Calculate velocity and acceleration
            velocity, acceleration = self.calculate_velocities_and_acceleration(
                landmarks, prev_landmarks, prev_velocity
            )

            # Store current as previous for next iteration
            prev_velocity = velocity if i > 0 else None
            prev_landmarks = landmarks

            feature = EnhancedGestureFeature(
                pose_landmarks=landmarks,
                optical_flow=optical_flow,
                motion_history=motion_history,
                hog_features=hog_features,
                joint_angles=joint_angles,
                trajectories=trajectories,
                velocity_vectors=[velocity] if i > 0 else [],
                acceleration=[acceleration],
                temporal_keypoints=[]  # Would be computed based on gesture segmentation
            )

            features.append(feature)

        return features

    def add_to_database(self, name: str, gesture_features: List[EnhancedGestureFeature]):
        """Add a gesture to the database"""
        self.gesture_database[name] = gesture_features

    def compare_gestures(self, input_features: List[EnhancedGestureFeature],
                         db_features: List[EnhancedGestureFeature]) -> float:
        """
        Compare two gesture sequences and return similarity score
        """
        if not input_features or not db_features:
            return 0.0

        # Normalize sequence lengths
        min_len = min(len(input_features), len(db_features))
        input_subset = input_features[:min_len]
        db_subset = db_features[:min_len]

        similarities = []

        for i in range(min_len):
            input_feat = input_subset[i]
            db_feat = db_subset[i]

            # Calculate similarity for various aspects
            pose_sim = self._pose_similarity(input_feat.pose_landmarks, db_feat.pose_landmarks)
            angle_sim = self._angle_similarity(input_feat.joint_angles, db_feat.joint_angles)
            trajectory_sim = self._trajectory_similarity(input_feat.trajectories, db_feat.trajectories)
            motion_sim = self._motion_similarity(input_feat.velocity_vectors, db_feat.velocity_vectors)

            # Weighted average with emphasis on poses and trajectories for tennis
            total_sim = (0.4 * pose_sim + 0.2 * angle_sim + 0.25 * trajectory_sim + 0.15 * motion_sim)
            similarities.append(total_sim)

        # Return average similarity across all frames
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _pose_similarity(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """Calculate similarity between two sets of pose landmarks"""
        if landmarks1.size == 0 or landmarks2.size == 0:
            return 0.0

        # Calculate normalized Euclidean distance
        diff = landmarks1 - landmarks2
        distances = np.linalg.norm(diff, axis=1)
        avg_distance = np.mean(distances)

        # Convert to similarity score (0-1, where 1 is most similar)
        max_expected_distance = 0.3  # Tuned parameter for tennis movements
        similarity = max(0, 1 - avg_distance / max_expected_distance)
        return similarity

    def _angle_similarity(self, angles1: List[float], angles2: List[float]) -> float:
        """Calculate similarity between two sets of angles"""
        if not angles1 or not angles2:
            return 0.0

        min_len = min(len(angles1), len(angles2))
        angles1 = angles1[:min_len]
        angles2 = angles2[:min_len]

        diffs = [abs(a1 - a2) for a1, a2 in zip(angles1, angles2)]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0.0

        # Convert to similarity (angles difference up to 20 degrees considered perfect)
        max_expected_diff = 30.0  # Increased for tennis movements
        similarity = max(0, 1 - avg_diff / max_expected_diff)
        return similarity

    def _trajectory_similarity(self, trajectories1: List[List[Tuple[float, float]]],
                              trajectories2: List[List[Tuple[float, float]]]) -> float:
        """Calculate similarity between two trajectory sets"""
        if not trajectories1 or not trajectories2:
            return 0.0

        # Calculate similarity for each trajectory
        total_similarity = 0.0
        count = 0

        for t1, t2 in zip(trajectories1, trajectories2):
            if t1 and t2:
                # Take min length for comparison
                min_len = min(len(t1), len(t2))
                if min_len > 0:
                    t1 = t1[-min_len:]  # Last min_len points
                    t2 = t2[-min_len:]

                    # Calculate average distance between corresponding points
                    distances = [euclidean(p1, p2) for p1, p2 in zip(t1, t2)]
                    avg_distance = sum(distances) / len(distances)

                    # Convert to similarity
                    max_expected_distance = 0.3  # Tuned for normalized coordinates
                    similarity = max(0, 1 - avg_distance / max_expected_distance)
                    total_similarity += similarity
                    count += 1

        return total_similarity / count if count > 0 else 0.0

    def _motion_similarity(self, velocities1: List[np.ndarray], velocities2: List[np.ndarray]) -> float:
        """Calculate similarity between motion patterns"""
        if not velocities1 or not velocities2:
            return 0.0

        min_len = min(len(velocities1), len(velocities2))
        velocities1 = velocities1[:min_len]
        velocities2 = velocities2[:min_len]

        total_similarity = 0.0
        count = 0

        for v1, v2 in zip(velocities1, velocities2):
            if v1.size > 0 and v2.size > 0:
                # Calculate cosine similarity between velocity vectors
                v1_flat = v1.flatten()
                v2_flat = v2.flatten()

                if np.linalg.norm(v1_flat) > 0 and np.linalg.norm(v2_flat) > 0:
                    dot_product = np.dot(v1_flat, v2_flat)
                    norms = np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat)
                    if norms > 0:
                        similarity = abs(dot_product / norms)  # Cosine similarity
                        total_similarity += similarity
                        count += 1

        return total_similarity / count if count > 0 else 0.0

    def find_best_match(self, input_video_path: str) -> Dict:
        """
        Find the best matching tennis player gesture from the database
        """
        # Extract features from input video
        input_features = self.extract_features_from_video(input_video_path)

        if not input_features:
            return {
                'best_match': None,
                'similarity_score': 0.0,
                'differences': [],
                'recommendations': []
            }

        # Compare with all database entries
        best_match = None
        best_similarity = 0.0

        for name, db_features in self.gesture_database.items():
            similarity = self.compare_gestures(input_features, db_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        # Calculate differences if a match was found
        differences = []
        recommendations = []
        if best_match:
            db_features = self.gesture_database[best_match]
            differences = self._calculate_differences(input_features, db_features)
            recommendations = self._generate_recommendations(input_features, db_features)

        return {
            'best_match': best_match,
            'similarity_score': best_similarity,
            'differences': differences,
            'recommendations': recommendations
        }

    def extract_features_from_video(self, video_path: str) -> List[EnhancedGestureFeature]:
        """
        Extract gesture features from a video file
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Capture all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # Extract features from frames
        return self.extract_enhanced_gesture_features(frames)

    def _calculate_differences(self, input_features: List[EnhancedGestureFeature],
                              db_features: List[EnhancedGestureFeature]) -> List[Dict]:
        """
        Calculate specific differences between input and database gestures
        """
        differences = []
        min_len = min(len(input_features), len(db_features))

        for i in range(min_len):
            input_feat = input_features[i]
            db_feat = db_features[i]

            # Find pose landmark differences
            pose_diff = input_feat.pose_landmarks - db_feat.pose_landmarks
            avg_pose_diff = np.mean(np.abs(pose_diff), axis=0) if pose_diff.size > 0 else np.zeros(2)

            # Find angle differences
            angle_diffs = []
            if input_feat.joint_angles and db_feat.joint_angles:
                min_angle_len = min(len(input_feat.joint_angles), len(db_feat.joint_angles))
                angle_diffs = [abs(a1 - a2) for a1, a2 in
                              zip(input_feat.joint_angles[:min_angle_len],
                                  db_feat.joint_angles[:min_angle_len])]

            # Find trajectory differences
            traj_diffs = []
            for t1, t2 in zip(input_feat.trajectories, db_feat.trajectories):
                if t1 and t2:
                    min_traj_len = min(len(t1), len(t2))
                    if min_traj_len > 0:
                        t1_short = t1[-min_traj_len:]
                        t2_short = t2[-min_traj_len:]
                        distances = [euclidean(p1, p2) for p1, p2 in zip(t1_short, t2_short)]
                        avg_traj_diff = sum(distances) / len(distances)
                        traj_diffs.append(avg_traj_diff)

            diff_info = {
                'frame_index': i,
                'pose_deviation': avg_pose_diff.tolist(),
                'angle_differences': angle_diffs,
                'trajectory_differences': traj_diffs,
                'velocity_difference': self._calculate_velocity_diff(
                    input_feat.velocity_vectors, db_feat.velocity_vectors)
            }

            differences.append(diff_info)

        return differences

    def _calculate_velocity_diff(self, velocities1: List[np.ndarray],
                               velocities2: List[np.ndarray]) -> float:
        """Calculate velocity difference"""
        if not velocities1 or not velocities2:
            return 0.0

        min_len = min(len(velocities1), len(velocities2))
        vel1 = velocities1[:min_len]
        vel2 = velocities2[:min_len]

        total_diff = 0.0
        for v1, v2 in zip(vel1, vel2):
            if v1.size > 0 and v2.size > 0:
                diff = v1 - v2
                total_diff += np.mean(np.abs(diff))

        return total_diff / min_len if min_len > 0 else 0.0

    def _generate_recommendations(self, input_features: List[EnhancedGestureFeature],
                                 db_features: List[EnhancedGestureFeature]) -> List[str]:
        """
        Generate recommendations for improving technique based on comparison
        """
        recommendations = []

        if not input_features or not db_features:
            return ["Unable to generate recommendations: No features to compare"]

        # Analyze differences to generate specific recommendations
        # For example, if the input has different angles compared to pro
        if len(input_features) > 0 and len(db_features) > 0:
            input_angles = input_features[0].joint_angles if input_features[0].joint_angles else []
            db_angles = db_features[0].joint_angles if db_features[0].joint_angles else []

            if input_angles and db_angles:
                for i, (inp_ang, db_ang) in enumerate(zip(input_angles[:5], db_angles[:5])):
                    diff = abs(inp_ang - db_ang)
                    if diff > 15:  # If difference is significant
                        joint_names = ['Left shoulder-elbow-wrist', 'Right shoulder-elbow-wrist',
                                     'Left hip-knee-ankle', 'Right hip-knee-ankle', 'Left shoulder-hip-knee']
                        recommendations.append(f"Adjust your {joint_names[i] if i < len(joint_names) else 'joint'} angle "
                                             f"(currently {inp_ang:.1f}° vs pro {db_ang:.1f}°)")

        # Add general recommendations
        if not recommendations:
            recommendations.extend([
                "Focus on maintaining consistent elbow positioning during forehand",
                "Try to replicate the professional's follow-through trajectory",
                "Adjust shoulder rotation timing to match the professional",
                "Work on hip-knee-ankle alignment during stance"
            ])

        return recommendations

    def save_database(self, filepath: str):
        """Save the gesture database to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.gesture_database, f)

    def load_database(self, filepath: str):
        """Load the gesture database from a file"""
        with open(filepath, 'rb') as f:
            self.gesture_database = pickle.load(f)

    def reset(self):
        """Reset the analyzer state for processing a new video.

        This clears cached landmarks and closes the Pose instance
        to ensure clean state between videos.
        """
        self.prev_landmarks = None
        self.prev_landmarks_confidence = 0.0
        self.landmark_velocity = None

        # Close MediaPipe Pose if using it
        if not self.use_opencv_pose and MEDIAPIPE_LEGACY and hasattr(self, 'pose'):
            self.pose.close()
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def close(self):
        """Properly close the analyzer and release resources."""
        if not self.use_opencv_pose and MEDIAPIPE_LEGACY and hasattr(self, 'pose'):
            self.pose.close()


def create_enhanced_sample_database(analyzer: EnhancedTennisGestureAnalyzer):
    """
    Create an enhanced sample database with gestures from famous tennis players.

    Uses realistic MediaPipe Pose landmark structure (33 landmarks) with
    tennis-specific pose patterns for different stroke types.
    """
    # MediaPipe Pose has 33 landmarks:
    # 0: Nose, 1-4: Eyes/Ears, 5-10: Mouth/Lips, 11-12: Shoulders, 13-14: Elbows,
    # 15-16: Wrists, 17-20: Fingers, 21-22: Hips, 23-24: Hips (lower),
    # 25-26: Knees, 27-28: Ankles, 29-32: Heels/Toes

    def create_tennis_pose(stroke_type: str, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create realistic tennis pose landmarks for a given stroke phase."""
        landmarks = np.zeros((33, 2), dtype=np.float32)

        # Normalize frame index to stroke phase (0-1)
        phase = frame_idx / total_frames

        if stroke_type == "forehand":
            # Simulate a forehand stroke: preparation -> contact -> follow-through
            # Head (relatively stable)
            landmarks[0] = [0.5, 0.25 + 0.02 * np.sin(phase * np.pi)]  # Nose

            # Shoulders (rotating during stroke)
            shoulder_rotation = 0.15 * np.sin(phase * np.pi)
            landmarks[11] = [0.42 - shoulder_rotation, 0.35]  # Left shoulder
            landmarks[12] = [0.58 + shoulder_rotation, 0.35]  # Right shoulder

            # Elbows (bending during swing)
            elbow_bend = 0.3 * (1 - phase) if phase < 0.5 else 0.15
            landmarks[13] = [0.35 - 0.1 * phase, 0.45 + elbow_bend]  # Left elbow
            landmarks[14] = [0.65 + 0.2 * phase, 0.45 + elbow_bend]  # Right elbow (swinging)

            # Wrists (racquet hand follows through)
            if phase < 0.3:  # Preparation
                landmarks[15] = [0.30, 0.55]  # Left wrist
                landmarks[16] = [0.70, 0.55]  # Right wrist
            elif phase < 0.6:  # Contact
                landmarks[15] = [0.32, 0.58]
                landmarks[16] = [0.68, 0.58 + 0.1 * (phase - 0.3)]
            else:  # Follow-through
                landmarks[15] = [0.35, 0.60]
                landmarks[16] = [0.65 + 0.15 * (phase - 0.6) * 3, 0.55 - 0.1 * (phase - 0.6) * 3]

            # Hips (stable base)
            landmarks[23] = [0.43, 0.55]  # Left hip
            landmarks[24] = [0.57, 0.55]  # Right hip

            # Knees (slight bend)
            landmarks[25] = [0.42, 0.70]  # Left knee
            landmarks[26] = [0.58, 0.70]  # Right knee

            # Ankles
            landmarks[27] = [0.42, 0.85]  # Left ankle
            landmarks[28] = [0.58, 0.85]  # Right ankle

        elif stroke_type == "serve":
            # Simulate a serve: toss -> trophy -> contact -> follow-through
            toss_phase = min(phase * 2, 1.0)

            # Head (looking up at ball)
            landmarks[0] = [0.5, 0.22 + 0.03 * toss_phase]

            # Shoulders (trophy position)
            landmarks[11] = [0.42, 0.35 + 0.05 * toss_phase]  # Left shoulder (rising)
            landmarks[12] = [0.58, 0.38]  # Right shoulder

            # Elbows (trophy elbow up)
            landmarks[13] = [0.38, 0.42 + 0.15 * toss_phase]  # Left elbow (up for toss)
            landmarks[14] = [0.65, 0.40 + 0.2 * (1 - toss_phase)]  # Right elbow (cocked)

            # Wrists
            landmarks[15] = [0.35, 0.35 + 0.25 * toss_phase]  # Left wrist (toss hand)
            landmarks[16] = [0.70, 0.35 + 0.15 * (1 - toss_phase)]  # Right wrist (racquet back)

            # Lower body (knee bend)
            landmarks[23] = [0.43, 0.55]
            landmarks[24] = [0.57, 0.55]
            landmarks[25] = [0.42, 0.68 + 0.05 * (1 - toss_phase)]  # Left knee (bending)
            landmarks[26] = [0.58, 0.68 + 0.05 * (1 - toss_phase)]  # Right knee
            landmarks[27] = [0.42, 0.83]
            landmarks[28] = [0.58, 0.83]

        else:  # Default/neutral stance
            landmarks[0] = [0.5, 0.25]
            landmarks[11] = [0.42, 0.35]
            landmarks[12] = [0.58, 0.35]
            landmarks[13] = [0.35, 0.45]
            landmarks[14] = [0.65, 0.45]
            landmarks[15] = [0.30, 0.55]
            landmarks[16] = [0.70, 0.55]
            landmarks[23] = [0.43, 0.55]
            landmarks[24] = [0.57, 0.55]
            landmarks[25] = [0.42, 0.70]
            landmarks[26] = [0.58, 0.70]
            landmarks[27] = [0.42, 0.85]
            landmarks[28] = [0.58, 0.85]

        # Fill in remaining landmarks with reasonable defaults
        # Eyes (1-4)
        landmarks[1] = [0.48, 0.23]  # Left eye
        landmarks[2] = [0.52, 0.23]  # Right eye
        landmarks[3] = [0.47, 0.25]  # Left ear
        landmarks[4] = [0.53, 0.25]  # Right ear

        # Mouth area (5-10) - simplified
        for i in range(5, 11):
            landmarks[i] = [0.47 + 0.012 * (i - 5), 0.28]

        # Fingers (17-20) - near wrists
        landmarks[17] = landmarks[15] + [0.02, 0.01]  # Left hand
        landmarks[18] = landmarks[15] + [0.03, 0.02]
        landmarks[19] = landmarks[16] + [-0.02, 0.01]  # Right hand
        landmarks[20] = landmarks[16] + [-0.03, 0.02]

        # Hip pointers (21-22)
        landmarks[21] = [0.45, 0.50]
        landmarks[22] = [0.55, 0.50]

        # Feet (29-32)
        landmarks[29] = [0.40, 0.90]  # Left heel
        landmarks[30] = [0.44, 0.92]  # Left toe
        landmarks[31] = [0.56, 0.92]  # Right toe
        landmarks[32] = [0.60, 0.90]  # Right heel

        return landmarks

    # Create Federer forehand (smooth, classic technique)
    federer_forehand = []
    for i in range(20):
        landmarks = create_tennis_pose("forehand", i, 20)
        # Add slight variation for realism
        landmarks += np.random.normal(0, 0.01, landmarks.shape)

        feature = EnhancedGestureFeature(
            pose_landmarks=landmarks,
            optical_flow=np.random.rand(480, 640, 2) * 0.1,
            motion_history=np.random.rand(480, 640) * 0.2,
            hog_features=np.random.rand(3780),
            joint_angles=[90.0 + np.random.normal(0, 2) for _ in range(12)],
            trajectories=[],  # Would be computed during extraction
            velocity_vectors=[np.random.rand(33, 2) * 0.05],
            acceleration=[np.random.uniform(0.1, 0.5)],
            temporal_keypoints=[]
        )
        federer_forehand.append(feature)

    # Create Nadal forehand (more extreme topspin motion)
    nadal_forehand = []
    for i in range(20):
        landmarks = create_tennis_pose("forehand", i, 20)
        # Nadal's more extreme wrist motion
        landmarks[16, 0] += 0.05 * np.sin(i / 20 * np.pi)  # More racquet hand movement
        landmarks += np.random.normal(0, 0.015, landmarks.shape)

        feature = EnhancedGestureFeature(
            pose_landmarks=landmarks,
            optical_flow=np.random.rand(480, 640, 2) * 0.12,
            motion_history=np.random.rand(480, 640) * 0.25,
            hog_features=np.random.rand(3780),
            joint_angles=[85.0 + np.random.normal(0, 3) for _ in range(12)],
            trajectories=[],
            velocity_vectors=[np.random.rand(33, 2) * 0.06],
            acceleration=[np.random.uniform(0.15, 0.6)],
            temporal_keypoints=[]
        )
        nadal_forehand.append(feature)

    # Create Serena serve (powerful, high toss)
    serena_serve = []
    for i in range(20):
        landmarks = create_tennis_pose("serve", i, 20)
        landmarks += np.random.normal(0, 0.01, landmarks.shape)

        feature = EnhancedGestureFeature(
            pose_landmarks=landmarks,
            optical_flow=np.random.rand(480, 640, 2) * 0.15,
            motion_history=np.random.rand(480, 640) * 0.3,
            hog_features=np.random.rand(3780),
            joint_angles=[95.0 + np.random.normal(0, 2) for _ in range(12)],
            trajectories=[],
            velocity_vectors=[np.random.rand(33, 2) * 0.07],
            acceleration=[np.random.uniform(0.2, 0.7)],
            temporal_keypoints=[]
        )
        serena_serve.append(feature)

    # Add to database
    analyzer.add_to_database("Roger Federer - Forehand", federer_forehand)
    analyzer.add_to_database("Rafael Nadal - Forehand", nadal_forehand)
    analyzer.add_to_database("Serena Williams - Serve", serena_serve)


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = EnhancedTennisGestureAnalyzer()

    # Create enhanced sample database
    create_enhanced_sample_database(analyzer)

    print("Enhanced Tennis Gesture Analysis System Ready!")
    print(f"Database contains {len(analyzer.gesture_database)} gesture samples")

    # Example usage (this would normally be triggered by a user request)
    # analyzer.find_best_match("input_video.mp4")