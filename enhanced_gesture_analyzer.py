import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.spatial.distance import euclidean
import math


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
    def __init__(self):
        # Initialize OpenCV's pose estimator (using basic pose detection for compatibility)
        # In a real implementation, you'd integrate MediaPipe or another pose estimator
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
        """Calculate angles between key joints"""
        if len(landmarks) < 33:  # Not enough landmarks
            return [0.0] * 10  # Return default angles

        angles = []

        # Define triplets of joints to calculate angles for
        angle_triplets = [
            # Shoulder-elbow-wrist angles (for racquet swing analysis)
            (11, 13, 15),  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            (12, 14, 16),  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
            # Hip-knee-ankle angles (for stance analysis)
            (23, 25, 27),  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            (24, 26, 28),  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            # Knee flexion during stroke preparation
            (23, 25, 27),  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            (24, 26, 28),  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            # Elbow flexion during swing
            (11, 13, 15),  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            (12, 14, 16),  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
            # Shoulder abduction
            (11, 23, 25),  # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
            (12, 24, 26),  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE
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
            except:
                angles.append(0.0)  # Default if calculation fails

        return angles

    def extract_trajectories(self, frame_sequence: List[np.ndarray], current_idx: int, landmarks: np.ndarray) -> List[List[Tuple[float, float]]]:
        """Extract movement trajectories of key joints over recent frames"""
        trajectory_length = min(15, current_idx + 1)  # Track last 15 frames
        trajectories = []

        # Define key joints to track
        key_joint_indices = [15, 16, 11, 12, 13, 14]  # Wrists and shoulders

        for joint_idx in key_joint_indices:
            if joint_idx >= len(landmarks):
                continue

            trajectory = []
            start_frame_idx = max(0, current_idx - trajectory_length)

            for frame_idx in range(start_frame_idx, current_idx + 1):
                # Get landmarks from the specific frame
                frame_landmarks = self.extract_landmarks_from_frame(frame_sequence[frame_idx])

                if frame_landmarks is not None and joint_idx < len(frame_landmarks):
                    x, y = frame_landmarks[joint_idx][0], frame_landmarks[joint_idx][1]
                    trajectory.append((x, y))

            trajectories.append(trajectory)

        return trajectories

    def extract_landmarks_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Simulate landmark extraction for this implementation"""
        # In a real implementation, this would use MediaPipe or OpenPose
        # For now, we'll simulate landmarks based on detecting human figures
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simple person detection using background subtraction concepts
        # In reality, this would use a pose estimation model
        # For this demo, we'll simulate landmarks by detecting contours

        # Create a simple body outline simulation
        h, w = frame.shape[:2]

        # Simulate landmarks as approximate body parts
        # This would be replaced by actual pose estimator
        landmarks = np.array([
            [w//2, h//3],      # Nose (approx)
            [w//2-20, h//3],   # Left eye
            [w//2+20, h//3],   # Right eye
            [w//2-30, h//3+10], # Left ear
            [w//2+30, h//3+10], # Right ear
            [w//2-40, h//2],   # Left shoulder
            [w//2+40, h//2],   # Right shoulder
            [w//2-60, h//2+50], # Left elbow
            [w//2+60, h//2+50], # Right elbow
            [w//2-70, h//2+100], # Left wrist
            [w//2+70, h//2+100], # Right wrist
            [w//2-40, h//2+120], # Left hip
            [w//2+40, h//2+120], # Right hip
            [w//2-50, h//2+180], # Left knee
            [w//2+50, h//2+180], # Right knee
            [w//2-50, h//2+240], # Left ankle
            [w//2+50, h//2+240], # Right ankle
        ], dtype=np.float32)

        # Normalize coordinates to [0, 1] range
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        return landmarks

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
        Extract enhanced features from a sequence of video frames
        """
        features = []
        prev_landmarks = None
        prev_velocity = None

        for i, frame in enumerate(frame_sequence):
            # Extract pose landmarks
            landmarks = self.extract_landmarks_from_frame(frame)

            if landmarks is None or len(landmarks) == 0:
                continue

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

            # Extract trajectories
            trajectories = self.extract_trajectories(frame_sequence, i, landmarks)

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


def create_enhanced_sample_database(analyzer: EnhancedTennisGestureAnalyzer):
    """
    Create an enhanced sample database with gestures from famous tennis players
    """
    # Generate more realistic sample data
    sample_gestures = []

    # Create sample gesture with realistic tennis movement patterns
    for _ in range(20):  # 20 frames for Federer forehand
        # Simulate tennis-specific poses
        landmarks = np.random.rand(16, 2) * 0.5 + 0.25  # Normalized coordinates

        # Create more structured landmarks for tennis
        landmarks[0] = [0.5, 0.3]  # Head
        landmarks[5] = [0.4, 0.4]  # Left shoulder
        landmarks[6] = [0.6, 0.4]  # Right shoulder
        landmarks[7] = [0.3, 0.5]  # Left elbow
        landmarks[8] = [0.7, 0.5]  # Right elbow
        landmarks[9] = [0.25, 0.6] # Left wrist
        landmarks[10] = [0.75, 0.6] # Right wrist
        # ... other landmarks

        feature = EnhancedGestureFeature(
            pose_landmarks=landmarks,
            optical_flow=np.random.rand(480, 640, 2) * 0.1,  # Simulated optical flow
            motion_history=np.random.rand(480, 640) * 0.2,    # Simulated motion history
            hog_features=np.random.rand(3780),               # Standard HOG features
            joint_angles=[90.0, 85.0, 75.0, 80.0, 95.0, 70.0, 85.0, 90.0, 80.0, 88.0],
            trajectories=[[ (0.1, 0.2), (0.15, 0.25), (0.2, 0.3) ]],  # Example trajectory
            velocity_vectors=[np.random.rand(16, 2) * 0.05],
            acceleration=[np.random.uniform(0.1, 0.5)],
            temporal_keypoints=[]
        )
        sample_gestures.append(feature)

    # Add to database with tennis-specific names
    analyzer.add_to_database("Roger Federer - Forehand", sample_gestures)
    analyzer.add_to_database("Rafael Nadal - Forehand", sample_gestures.copy())
    analyzer.add_to_database("Serena Williams - Serve", sample_gestures.copy())


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = EnhancedTennisGestureAnalyzer()

    # Create enhanced sample database
    create_enhanced_sample_database(analyzer)

    print("Enhanced Tennis Gesture Analysis System Ready!")
    print(f"Database contains {len(analyzer.gesture_database)} gesture samples")

    # Example usage (this would normally be triggered by a user request)
    # analyzer.find_best_match("input_video.mp4")