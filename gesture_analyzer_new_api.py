import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import mediapipe as mp
from mediapipe.tasks import vision


@dataclass
class GestureFeature:
    """Represents features extracted from a tennis gesture"""
    landmarks: np.ndarray  # MediaPipe pose landmarks
    key_points: List[Tuple[int, float, float]]  # Key joint positions
    velocity_vectors: List[np.ndarray]  # Movement vectors between frames
    acceleration: List[float]  # Acceleration values
    angles: List[float]  # Joint angles
    trajectory: List[Tuple[float, float]]  # Movement path of key joints


class TennisGestureAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Pose - using the newer API
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create pose landmarker
        self.pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
                running_mode=VisionRunningMode.IMAGE  # Use IMAGE mode for frame-by-frame processing
            )
        )

        # Initialize the gesture database
        self.gesture_database = {}

    def extract_gesture_features(self, frame_sequence: List[np.ndarray]) -> List[GestureFeature]:
        """
        Extract features from a sequence of video frames
        """
        features = []

        for i, frame in enumerate(frame_sequence):
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose landmarks
            detection_result = self.pose_landmarker.detect(mp_image)

            # Check if landmarks were detected
            if detection_result.pose_landmarks:
                # Extract landmarks
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in detection_result.pose_landmarks[0]  # Take first person detected
                ])

                # Calculate key joint positions
                key_joints = [
                    ('left_shoulder', detection_result.pose_landmarks[0][11]),  # LEFT_SHOULDER index
                    ('right_shoulder', detection_result.pose_landmarks[0][12]),  # RIGHT_SHOULDER index
                    ('left_elbow', detection_result.pose_landmarks[0][13]),     # LEFT_ELBOW index
                    ('right_elbow', detection_result.pose_landmarks[0][14]),    # RIGHT_ELBOW index
                    ('left_wrist', detection_result.pose_landmarks[0][15]),     # LEFT_WRIST index
                    ('right_wrist', detection_result.pose_landmarks[0][16]),    # RIGHT_WRIST index
                    ('left_hip', detection_result.pose_landmarks[0][23]),       # LEFT_HIP index
                    ('right_hip', detection_result.pose_landmarks[0][24]),      # RIGHT_HIP index
                    ('left_knee', detection_result.pose_landmarks[0][25]),      # LEFT_KNEE index
                    ('right_knee', detection_result.pose_landmarks[0][26]),     # RIGHT_KNEE index
                    ('left_ankle', detection_result.pose_landmarks[0][27]),     # LEFT_ANKLE index
                    ('right_ankle', detection_result.pose_landmarks[0][28]),    # RIGHT_ANKLE index
                ]

                key_points = [(joint[0], joint[1].x, joint[1].y) for joint in key_joints]

                # Calculate angles between joints
                angles = self._calculate_joint_angles(detection_result.pose_landmarks[0])

                # Velocity and acceleration (for movement sequences)
                velocity_vectors = []
                acceleration = []

                if i > 0:
                    # Previous landmarks for calculating movement
                    prev_results = self._get_previous_landmarks(i-1, frame_sequence[:i])
                    if prev_results and prev_results.pose_landmarks:
                        prev_landmarks_list = prev_results.pose_landmarks[0]
                        prev_landmarks = np.array([
                            [lm.x, lm.y, lm.z] for lm in prev_landmarks_list
                        ])

                        # Calculate velocity vectors (change in position over time)
                        velocity = landmarks - prev_landmarks
                        velocity_vectors.append(velocity)

                        # Calculate acceleration if we have more than 2 frames
                        if i >= 2:
                            prev_prev_results = self._get_previous_landmarks(i-2, frame_sequence[:i-1])
                            if prev_prev_results and prev_prev_results.pose_landmarks:
                                prev_prev_landmarks_list = prev_prev_results.pose_landmarks[0]
                                prev_prev_landmarks = np.array([
                                    [lm.x, lm.y, lm.z] for lm in prev_prev_landmarks_list
                                ])
                                prev_velocity = prev_landmarks - prev_prev_landmarks
                                acc = velocity - prev_velocity
                                acceleration.append(np.linalg.norm(acc))

                # Extract trajectory (path of key joints over time)
                trajectory = self._extract_trajectory(frame_sequence, i)

                feature = GestureFeature(
                    landmarks=landmarks,
                    key_points=key_points,
                    velocity_vectors=velocity_vectors,
                    acceleration=acceleration,
                    angles=angles,
                    trajectory=trajectory
                )

                features.append(feature)

        return features

    def _get_previous_landmarks(self, frame_idx: int, sequence: List[np.ndarray]):
        """Helper to get landmarks from a previous frame"""
        if frame_idx < len(sequence):
            frame = sequence[frame_idx]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            return self.pose_landmarker.detect(mp_image)
        return None

    def _calculate_joint_angles(self, landmarks) -> List[float]:
        """Calculate angles between key joints"""
        angles = []

        # Define triplets of joints to calculate angles for (using MediaPipe indices)
        angle_triplets = [
            # Shoulder-elbow-wrist angles
            (11, 13, 15),  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            (12, 14, 16),  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

            # Hip-knee-ankle angles
            (23, 25, 27),  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            (24, 26, 28),  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        ]

        for triplet in angle_triplets:
            p1_idx, p2_idx, p3_idx = triplet
            try:
                # Convert to numpy arrays
                point1 = np.array([landmarks[p1_idx].x, landmarks[p1_idx].y])
                point2 = np.array([landmarks[p2_idx].x, landmarks[p2_idx].y])
                point3 = np.array([landmarks[p3_idx].x, landmarks[p3_idx].y])

                # Calculate vectors
                v1 = point1 - point2
                v2 = point3 - point2

                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(np.degrees(angle))
            except:
                angles.append(0.0)  # Default if calculation fails

        return angles

    def _extract_trajectory(self, frame_sequence: List[np.ndarray], current_idx: int) -> List[Tuple[float, float]]:
        """Extract trajectory of key joints over recent frames"""
        trajectory_length = min(10, current_idx + 1)  # Track last 10 frames
        trajectory = []

        for i in range(max(0, current_idx - trajectory_length), current_idx + 1):
            frame = frame_sequence[i]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.pose_landmarker.detect(mp_image)

            if results.pose_landmarks:
                # Track wrist position (as example)
                wrist_x = results.pose_landmarks[0][16].x  # RIGHT_WRIST index
                wrist_y = results.pose_landmarks[0][16].y  # RIGHT_WRIST index
                trajectory.append((wrist_x, wrist_y))

        return trajectory

    def add_to_database(self, name: str, gesture_features: List[GestureFeature]):
        """Add a gesture to the database"""
        self.gesture_database[name] = gesture_features

    def compare_gestures(self, input_features: List[GestureFeature],
                         db_features: List[GestureFeature]) -> float:
        """
        Compare two gesture sequences and return similarity score
        """
        if not input_features or not db_features:
            return 0.0

        # Compare using dynamic time warping (DTW) or simple distance metrics
        # For simplicity, we'll use a weighted average of multiple features

        # Normalize sequence lengths
        min_len = min(len(input_features), len(db_features))
        input_subset = input_features[:min_len]
        db_subset = db_features[:min_len]

        similarities = []

        for i in range(min_len):
            input_feat = input_subset[i]
            db_feat = db_subset[i]

            # Calculate similarity for various aspects
            landmark_sim = self._landmark_similarity(input_feat.landmarks, db_feat.landmarks)
            angle_sim = self._angle_similarity(input_feat.angles, db_feat.angles)
            trajectory_sim = self._trajectory_similarity(input_feat.trajectory, db_feat.trajectory)

            # Weighted average
            total_sim = (0.5 * landmark_sim + 0.3 * angle_sim + 0.2 * trajectory_sim)
            similarities.append(total_sim)

        # Return average similarity across all frames
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _landmark_similarity(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """Calculate similarity between two sets of landmarks"""
        # Calculate normalized Euclidean distance
        diff = landmarks1 - landmarks2
        distances = np.linalg.norm(diff, axis=1)
        avg_distance = np.mean(distances)

        # Convert to similarity score (0-1, where 1 is most similar)
        max_expected_distance = 0.5  # Tuned parameter
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
        max_expected_diff = 20.0
        similarity = max(0, 1 - avg_diff / max_expected_diff)
        return similarity

    def _trajectory_similarity(self, traj1: List[Tuple[float, float]], traj2: List[Tuple[float, float]]) -> float:
        """Calculate similarity between two trajectories"""
        if not traj1 or not traj2:
            return 0.0

        # Simple distance-based similarity
        min_len = min(len(traj1), len(traj2))
        traj1 = traj1[-min_len:]  # Take last min_len points
        traj2 = traj2[-min_len:]

        total_distance = 0.0
        for pt1, pt2 in zip(traj1, traj2):
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            dist = np.sqrt(dx*dx + dy*dy)
            total_distance += dist

        avg_distance = total_distance / min_len
        # Convert to similarity (max 0.5 normalized distance is perfect match)
        max_expected_distance = 0.5
        similarity = max(0, 1 - avg_distance / max_expected_distance)
        return similarity

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

    def extract_features_from_video(self, video_path: str) -> List[GestureFeature]:
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
        return self.extract_gesture_features(frames)

    def _calculate_differences(self, input_features: List[GestureFeature],
                              db_features: List[GestureFeature]) -> List[Dict]:
        """
        Calculate specific differences between input and database gestures
        """
        differences = []
        min_len = min(len(input_features), len(db_features))

        for i in range(min_len):
            input_feat = input_features[i]
            db_feat = db_features[i]

            # Find landmark differences
            landmark_diff = input_feat.landmarks - db_feat.landmarks
            avg_diff = np.mean(np.abs(landmark_diff), axis=0)

            # Find angle differences
            if input_feat.angles and db_feat.angles:
                angle_diffs = [abs(a1 - a2) for a1, a2 in
                               zip(input_feat.angles[:len(db_feat.angles)],
                                   db_feat.angles)]
            else:
                angle_diffs = []

            diff_info = {
                'frame_index': i,
                'landmark_deviation': avg_diff.tolist(),
                'angle_differences': angle_diffs,
                'trajectory_difference': self._calculate_trajectory_diff(
                    input_feat.trajectory, db_feat.trajectory)
            }

            differences.append(diff_info)

        return differences

    def _calculate_trajectory_diff(self, traj1: List[Tuple[float, float]],
                                  traj2: List[Tuple[float, float]]) -> float:
        """Calculate trajectory difference"""
        if not traj1 or not traj2:
            return 0.0

        min_len = min(len(traj1), len(traj2))
        traj1 = traj1[-min_len:]
        traj2 = traj2[-min_len:]

        total_diff = 0.0
        for pt1, pt2 in zip(traj1, traj2):
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            total_diff += np.sqrt(dx*dx + dy*dy)

        return total_diff / min_len if min_len > 0 else 0.0

    def _generate_recommendations(self, input_features: List[GestureFeature],
                                 db_features: List[GestureFeature]) -> List[str]:
        """
        Generate recommendations for improving technique based on comparison
        """
        recommendations = []

        if not input_features or not db_features:
            return ["Unable to generate recommendations: No features to compare"]

        # Example recommendations - would be enhanced in a real implementation
        recommendations.append("Focus on maintaining consistent elbow positioning during forehand")
        recommendations.append("Try to replicate the professional's follow-through trajectory")
        recommendations.append("Adjust shoulder rotation timing to match the professional")
        recommendations.append("Work on hip-knee-ankle alignment during stance")

        return recommendations

    def save_database(self, filepath: str):
        """Save the gesture database to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.gesture_database, f)

    def load_database(self, filepath: str):
        """Load the gesture database from a file"""
        with open(filepath, 'rb') as f:
            self.gesture_database = pickle.load(f)


def create_sample_database(analyzer: TennisGestureAnalyzer):
    """
    Create a sample database with gestures from famous tennis players
    In a real application, these would come from actual gesture data
    """
    # Sample data - in reality these would be extracted from videos of tennis players
    sample_gestures = [
        # Federer forehand
        [GestureFeature(
            landmarks=np.random.rand(33, 3) * 0.5,
            key_points=[('shoulder', 0.2, 0.3), ('elbow', 0.3, 0.4)],
            velocity_vectors=[],
            acceleration=[],
            angles=[90.0, 85.0, 75.0],
            trajectory=[(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)]
        ) for _ in range(20)],

        # Nadal forehand
        [GestureFeature(
            landmarks=np.random.rand(33, 3) * 0.5 + 0.1,
            key_points=[('shoulder', 0.25, 0.35), ('elbow', 0.35, 0.45)],
            velocity_vectors=[],
            acceleration=[],
            angles=[95.0, 80.0, 70.0],
            trajectory=[(0.12, 0.22), (0.18, 0.28), (0.22, 0.32)]
        ) for _ in range(20)],

        # Williams serve
        [GestureFeature(
            landmarks=np.random.rand(33, 3) * 0.5 - 0.1,
            key_points=[('shoulder', 0.18, 0.28), ('elbow', 0.28, 0.38)],
            velocity_vectors=[],
            acceleration=[],
            angles=[85.0, 90.0, 80.0],
            trajectory=[(0.08, 0.18), (0.12, 0.22), (0.18, 0.28)]
        ) for _ in range(15)],
    ]

    # Add to database
    analyzer.add_to_database("Roger Federer - Forehand", sample_gestures[0])
    analyzer.add_to_database("Rafael Nadal - Forehand", sample_gestures[1])
    analyzer.add_to_database("Serena Williams - Serve", sample_gestures[2])


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TennisGestureAnalyzer()

    # Create sample database
    create_sample_database(analyzer)

    print("Tennis Gesture Analysis System Ready!")
    print(f"Database contains {len(analyzer.gesture_database)} gesture samples")

    # Example usage (this would normally be triggered by a user request)
    # analyzer.find_best_match("input_video.mp4")