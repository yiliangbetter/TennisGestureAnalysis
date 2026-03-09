import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import pickle


@dataclass
class GestureFeature:
    """Represents features extracted from a tennis gesture"""
    landmarks: np.ndarray  # Pose landmarks (simulated)
    key_points: List[Tuple[int, float, float]]  # Key joint positions
    velocity_vectors: List[np.ndarray]  # Movement vectors between frames
    acceleration: List[float]  # Acceleration values
    angles: List[float]  # Joint angles
    trajectory: List[Tuple[float, float]]  # Movement path of key joints


class TennisGestureAnalyzer:
    def __init__(self):
        # Initialize the gesture database
        self.gesture_database = {}

    def extract_gesture_features(self, frame_sequence: List[np.ndarray]) -> List[GestureFeature]:
        """
        Extract features from a sequence of video frames (simulated for now)
        In a real implementation, this would use MediaPipe or another pose estimation library
        """
        features = []

        for i, frame in enumerate(frame_sequence):
            # Simulate pose landmark extraction (in real implementation, this would use MediaPipe)
            # Here we simulate landmarks for a human pose
            landmarks = np.random.rand(33, 3) * 0.5  # Simulated landmarks

            # Calculate key joint positions (simulated)
            key_points = [
                ('left_shoulder', np.random.uniform(0.2, 0.4), np.random.uniform(0.3, 0.5)),
                ('right_shoulder', np.random.uniform(0.6, 0.8), np.random.uniform(0.3, 0.5)),
                ('left_elbow', np.random.uniform(0.1, 0.3), np.random.uniform(0.4, 0.6)),
                ('right_elbow', np.random.uniform(0.7, 0.9), np.random.uniform(0.4, 0.6)),
                ('left_wrist', np.random.uniform(0.0, 0.2), np.random.uniform(0.5, 0.7)),
                ('right_wrist', np.random.uniform(0.8, 1.0), np.random.uniform(0.5, 0.7)),
                ('left_hip', np.random.uniform(0.3, 0.5), np.random.uniform(0.5, 0.7)),
                ('right_hip', np.random.uniform(0.5, 0.7), np.random.uniform(0.5, 0.7)),
                ('left_knee', np.random.uniform(0.2, 0.4), np.random.uniform(0.7, 0.9)),
                ('right_knee', np.random.uniform(0.6, 0.8), np.random.uniform(0.7, 0.9)),
                ('left_ankle', np.random.uniform(0.2, 0.4), np.random.uniform(0.8, 1.0)),
                ('right_ankle', np.random.uniform(0.6, 0.8), np.random.uniform(0.8, 1.0)),
            ]

            # Calculate angles between joints (simulated)
            angles = [np.random.uniform(70, 120) for _ in range(10)]  # Simulated angles

            # Velocity and acceleration (simulated)
            velocity_vectors = [np.random.rand(33, 3) * 0.1] if i > 0 else []  # Simulated velocity
            acceleration = [np.random.uniform(0, 0.5)] if i > 1 else []  # Simulated acceleration

            # Extract trajectory (simulated)
            trajectory = [(np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)) for _ in range(min(i+1, 5))]

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