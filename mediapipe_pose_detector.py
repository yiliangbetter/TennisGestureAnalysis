#!/usr/bin/env python3
"""
MediaPipe Pose Detector using the Tasks API (MediaPipe 0.10+)

This uses the new MediaPipe Tasks API which is the recommended approach
for MediaPipe 0.10 and later versions.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import BaseOptions
from typing import Optional, Tuple, List
import os


class MediaPipePoseDetector:
    """
    Pose detector using MediaPipe Tasks API.

    Uses the PoseLandmarker task for detecting 33 body landmarks.
    """

    # MediaPipe pose landmarks (33 total)
    # 0: nose
    # 1-10: left eye, right eye, left ear, right ear, etc.
    # 11-12: left shoulder, right shoulder
    # 13-14: left elbow, right elbow
    # 15-16: left wrist, right wrist
    # 17-18: left pinky, right pinky
    # 19-20: left index, right index
    # 21-22: left thumb, right thumb
    # 23-24: left hip, right hip
    # 25-26: left knee, right knee
    # 27-28: left ankle, right ankle
    # 29-30: left heel, right heel
    # 31-32: left foot index, right foot index

    def __init__(self, model_path: Optional[str] = None,
                 min_pose_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe pose detector.

        Args:
            model_path: Path to the pose landmarker model (optional)
            min_pose_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.min_pose_confidence = min_pose_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Create pose landmarker options
        base_options = BaseOptions(
            model_asset_path=model_path if model_path else ''
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Create the pose landmarker
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect_pose(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect pose landmarks in a frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Tuple of (landmarks, confidence)
            landmarks: 33x2 array of normalized (x,y) coordinates, or None if no pose detected
            confidence: Overall detection confidence
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose
        detection_result = self.detector.detect(mp_image)

        # Check if any poses were detected
        if not detection_result.pose_landmarks:
            return None, 0.0

        # Get the first detected pose
        pose_landmarks = detection_result.pose_landmarks[0]

        # Convert to numpy array (33 landmarks, x and y coordinates)
        landmarks = np.zeros((33, 2), dtype=np.float32)
        confidences = []

        for i, landmark in enumerate(pose_landmarks):
            landmarks[i] = [landmark.x, landmark.y]
            confidences.append(landmark.visibility)

        # Calculate overall confidence as average visibility
        overall_confidence = np.mean(confidences) if confidences else 0.0

        return landmarks, overall_confidence

    def detect_pose_video(self, video_path: str,
                         sample_interval: int = 5) -> List[Tuple[np.ndarray, float, int]]:
        """
        Detect poses in a video file.

        Args:
            video_path: Path to video file
            sample_interval: Process every Nth frame (for performance)

        Returns:
            List of (landmarks, confidence, frame_number) tuples
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every Nth frame
            if frame_count % sample_interval == 0:
                landmarks, confidence = self.detect_pose(frame)
                if landmarks is not None:
                    results.append((landmarks, confidence, frame_count))

            frame_count += 1

        cap.release()
        return results

    def close(self):
        """Release resources."""
        if self.detector:
            self.detector.close()


def create_mediapipe_detector(model_path: Optional[str] = None,
                               min_confidence: float = 0.5) -> MediaPipePoseDetector:
    """
    Factory function to create a MediaPipe pose detector.

    Args:
        model_path: Optional path to custom model
        min_confidence: Minimum detection confidence

    Returns:
        Configured MediaPipePoseDetector instance
    """
    return MediaPipePoseDetector(
        model_path=model_path,
        min_pose_confidence=min_confidence,
        min_tracking_confidence=min_confidence
    )


if __name__ == "__main__":
    # Test the detector
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mediapipe_pose_detector.py <image_or_video>")
        sys.exit(1)

    detector = create_mediapipe_detector()
    input_path = sys.argv[1]

    # Check if image or video
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    if is_video:
        results = detector.detect_pose_video(input_path)
        print(f"Detected {len(results)} poses in video")
        for landmarks, conf, frame_num in results[:5]:
            print(f"  Frame {frame_num}: confidence={conf:.2f}")
    else:
        frame = cv2.imread(input_path)
        if frame is not None:
            landmarks, confidence = detector.detect_pose(frame)
            if landmarks is not None:
                print(f"Detected pose with confidence {confidence:.2f}")
                print(f"Nose position: {landmarks[0]}")
                print(f"Left shoulder: {landmarks[11]}")
                print(f"Right shoulder: {landmarks[12]}")
            else:
                print("No pose detected")
        else:
            print(f"Could not load image: {input_path}")

    detector.close()
