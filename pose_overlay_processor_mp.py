#!/usr/bin/env python3
"""
Pose Overlay Processor for Tennis Gesture Analysis - MediaPipe Only Version

This script processes tennis videos using MediaPipe Pose Landmarker for
both person detection and pose landmark extraction. No traditional CV
motion detection is used.

Key changes from original version:
- Removed PersonDetector class (CV-based motion detection)
- MediaPipe PoseLandmarker provides both person detection AND landmarks
- Person bbox is derived from landmark positions
- Much more robust - no false positives from ceiling lights, rackets, etc.

Usage:
    python pose_overlay_processor_mp.py <input_video> [output_video]

Example:
    python pose_overlay_processor_mp.py SampleInputVideos.mp4 SampleInputVideos_MediaPipeOnly.mp4
"""

import cv2
import numpy as np
import argparse
import sys
import os
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime

# Configure logging to file and console
def setup_logging(output_dir: str = "logs"):
    """Setup logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"pose_processor_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# MediaPipe imports - use new tasks API
try:
    import mediapipe as mp

    # Access vision module through mp.tasks.vision (works in MP 0.10.x)
    _vision = mp.tasks.vision
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = _vision.PoseLandmarker
    PoseLandmarkerOptions = _vision.PoseLandmarkerOptions
    RunningMode = _vision.RunningMode
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe tasks API: Available")
except ImportError as e:
    logger.error(f"MediaPipe tasks API: Not available - {e}")
    mp = None
    MEDIAPIPE_AVAILABLE = False
    PoseLandmarker = None
    PoseLandmarkerOptions = None
    BaseOptions = None
    RunningMode = None


class MediaPipePersonDetector:
    """
    Detect person and extract pose landmarks using MediaPipe Pose Landmarker.

    This is a one-stage detector - MediaPipe internally handles both:
    1. Person detection (finding people in the frame)
    2. Landmark extraction (33 body keypoints)

    Advantages over CV-based motion detection:
    - No false positives from ceiling lights, rackets, or other motion
    - Works even when person is still (no motion required)
    - Handles multiple people and selects the most prominent
    - No hand-tuned thresholds or heuristics
    - Robust to lighting changes, camera angles, occlusion
    """

    def __init__(self, model_path: str = "pose_landmarker_heavy.task", num_poses: int = 1):
        """
        Initialize MediaPipe Pose Landmarker.

        Args:
            model_path: Path to the .task model file
            num_poses: Number of people to detect (1 = most prominent person)
        """
        self.pose_landmarker = None
        self.num_poses = num_poses
        self.frame_count = 0
        self.detection_count = 0

        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe not available - detector will not function")
            return

        # Find model file
        if not os.path.exists(model_path):
            # Try common locations
            possible_paths = [
                model_path,
                os.path.join(os.path.dirname(__file__), model_path),
                os.path.join(os.getcwd(), model_path),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                logger.error(f"Model file not found: {model_path}")
                return

        try:
            # Configure MediaPipe
            base_options = BaseOptions(model_asset_path=model_path)
            options = PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_poses=num_poses,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.pose_landmarker = PoseLandmarker.create_from_options(options)
            logger.info(f"MediaPipe PoseLandmarker initialized with model: {model_path}")
            logger.info(f"Num poses: {num_poses}, Detection confidence: 0.5")

        except Exception as e:
            logger.error(f"Failed to initialize PoseLandmarker: {e}")
            self.pose_landmarker = None

    def detect_person_and_landmarks(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int, float]], Optional[np.ndarray]]:
        """
        Detect person and extract pose landmarks in a single step.

        Args:
            frame: BGR image frame

        Returns:
            Tuple of:
            - Person bbox: (x, y, w, h, confidence) or None
            - Landmarks: (33, 3) array of [x, y, visibility] or None
        """
        self.frame_count += 1

        if self.pose_landmarker is None:
            return None, None

        height, width = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect pose
        result = self.pose_landmarker.detect(mp_image)

        # Check if any pose was detected
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            logger.debug(f"Frame {self.frame_count}: No pose detected")
            return None, None

        # Get the first (most prominent) pose
        pose_landmarks = result.pose_landmarks[0]
        pose_world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

        self.detection_count += 1

        # Convert landmarks to numpy array (33 landmarks x 3 values: x, y, visibility)
        landmarks_array = np.zeros((len(pose_landmarks), 3), dtype=np.float32)
        for i, landmark in enumerate(pose_landmarks):
            landmarks_array[i] = [landmark.x, landmark.y, landmark.visibility]

        # Compute bounding box from landmarks
        bbox = self._landmarks_to_bbox(landmarks_array, width, height)

        if bbox:
            logger.debug(f"Frame {self.frame_count}: Detected person at {bbox[0]}, {bbox[1]}, {bbox[2]}x{bbox[3]}, conf={bbox[4]:.2f}")

        return bbox, landmarks_array

    def _landmarks_to_bbox(
        self,
        landmarks: np.ndarray,
        img_width: int,
        img_height: int,
        padding: float = 0.1
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Compute bounding box from pose landmarks.

        Args:
            landmarks: (33, 3) array of [x, y, visibility] in normalized coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
            padding: Padding fraction around the person (default 10%)

        Returns:
            (x, y, w, h, confidence) or None if insufficient landmarks
        """
        # Filter landmarks by visibility (only use visible landmarks)
        visible_mask = landmarks[:, 2] > 0.3  # visibility threshold
        visible_landmarks = landmarks[visible_mask]

        if len(visible_landmarks) < 5:  # Need at least 5 visible landmarks
            logger.debug(f"Insufficient visible landmarks: {len(visible_landmarks)}")
            return None

        # Get bounding box from visible landmarks
        x_coords = visible_landmarks[:, 0] * img_width
        y_coords = visible_landmarks[:, 1] * img_height

        # Calculate average visibility as confidence
        avg_visibility = np.mean(visible_landmarks[:, 2])

        # Add padding
        x_min = max(0, np.min(x_coords) - padding * np.ptp(x_coords))
        x_max = min(img_width, np.max(x_coords) + padding * np.ptp(x_coords))
        y_min = max(0, np.min(y_coords) - padding * np.ptp(y_coords))
        y_max = min(img_height, np.max(y_coords) + padding * np.ptp(y_coords))

        x = int(x_min)
        y = int(y_min)
        w = int(x_max - x_min)
        h = int(y_max - y_min)

        # Validate bbox
        if w < 50 or h < 50:  # Minimum reasonable size
            logger.debug(f"Bbox too small: {w}x{h}")
            return None

        return (x, y, w, h, avg_visibility)

    def get_detection_stats(self) -> Dict:
        """Get detection statistics."""
        detection_rate = self.detection_count / max(self.frame_count, 1)
        return {
            'total_frames': self.frame_count,
            'detected_frames': self.detection_count,
            'detection_rate': detection_rate,
        }


class PoseOverlayProcessor:
    """Process videos and overlay pose detection landmarks."""

    # Color palette for different body parts (BGR format)
    COLORS = {
        'nose': (0, 0, 255),           # Red
        'left_eye': (0, 128, 255),     # Orange
        'right_eye': (0, 128, 255),    # Orange
        'left_ear': (0, 128, 255),     # Orange
        'right_ear': (0, 128, 255),    # Orange
        'left_shoulder': (255, 0, 0),  # Blue
        'right_shoulder': (255, 0, 0), # Blue
        'left_elbow': (255, 0, 0),     # Blue
        'right_elbow': (255, 0, 0),    # Blue
        'left_wrist': (255, 0, 255),   # Magenta
        'right_wrist': (255, 0, 255),  # Magenta
        'left_hip': (0, 255, 0),       # Green
        'right_hip': (0, 255, 0),      # Green
        'left_knee': (0, 255, 0),      # Green
        'right_knee': (0, 255, 0),     # Green
        'left_ankle': (128, 0, 255),   # Purple
        'right_ankle': (128, 0, 255),  # Purple
        'torso': (128, 128, 0),        # Cyan
        'generic': (255, 255, 255)     # White
    }

    # Landmark connections (MediaPipe format)
    CONNECTIONS = [
        # Head
        (0, 1, 'nose'),      # Nose to left eye
        (0, 2, 'nose'),      # Nose to right eye
        (1, 3, 'left_eye'),  # Left eye to left ear
        (2, 4, 'right_eye'), # Right eye to right ear

        # Torso
        (11, 12, 'torso'),   # Left shoulder to right shoulder
        (11, 23, 'torso'),   # Left shoulder to left hip
        (12, 24, 'torso'),   # Right shoulder to right hip
        (23, 24, 'torso'),   # Left hip to right hip

        # Left arm
        (11, 13, 'left_shoulder'),   # Shoulder to elbow
        (13, 15, 'left_elbow'),      # Elbow to wrist

        # Right arm
        (12, 14, 'right_shoulder'),  # Shoulder to elbow
        (14, 16, 'right_elbow'),     # Elbow to wrist

        # Left leg
        (23, 25, 'left_hip'),        # Hip to knee
        (25, 27, 'left_knee'),       # Knee to ankle

        # Right leg
        (24, 26, 'right_hip'),       # Hip to knee
        (26, 28, 'right_knee'),      # Knee to ankle
    ]

    def __init__(self, show_pose: bool = True):
        """
        Initialize the processor.

        Args:
            show_pose: Whether to draw pose skeleton (True for full visualization,
                      False for bbox-only debug mode)
        """
        self.person_detector = MediaPipePersonDetector()
        self.show_pose = show_pose

    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: int = 0
    ):
        """
        Process video and save with overlays.

        Args:
            input_path: Path to input video
            output_path: Path to output video
            max_frames: Maximum frames to process (0 = all frames)
        """
        logger.info(f"Opening input video: {input_path}")
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_path}")
            sys.exit(1)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0  # Default fallback

        logger.info(f"Input video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error("Failed to create output video writer")
            sys.exit(1)

        logger.info(f"Output video: {output_path} (H.264 codec)")

        # Processing statistics
        total_processed = 0
        total_detected = 0
        bbox_positions = []

        frame_count = 0

        logger.info("Starting video processing...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            overlay = self.process_frame(frame, frame_count)
            out.write(overlay)

            # Track statistics
            total_processed += 1

            # Check if person was detected
            bbox = self.person_detector.detect_person_and_landmarks(frame)[0]
            if bbox:
                total_detected += 1
                bbox_positions.append((bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2))

            frame_count += 1

            # Progress reporting
            if frame_count % 50 == 0 or frame_count == total_frames:
                progress = (frame_count / max(total_frames, 1)) * 100
                logger.info(f"Processing: {progress:.1f}% ({frame_count}/{total_frames})")

            # Check max frames limit
            if max_frames > 0 and frame_count >= max_frames:
                break

        cap.release()
        out.release()

        # Final statistics
        detection_rate = total_detected / max(total_processed, 1) * 100

        # Calculate position variance
        position_variance = 0.0
        if len(bbox_positions) > 1:
            positions = np.array(bbox_positions)
            position_variance = np.mean(np.var(positions, axis=0))

        logger.info("")
        logger.info("=" * 60)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total frames:       {total_processed}")
        logger.info(f"Detected frames:    {total_detected}")
        logger.info(f"Detection rate:     {detection_rate:.1f}%")
        logger.info(f"Avg position var:   {position_variance:.5f}")
        logger.info("")
        logger.info(f"Output saved to: {output_path}")

        # Save detection stats to log
        stats = self.person_detector.get_detection_stats()
        logger.info("")
        logger.info("MediaPipe Detection Stats:")
        logger.info(f"  Total frames:     {stats['total_frames']}")
        logger.info(f"  Detected frames:  {stats['detected_frames']}")
        logger.info(f"  Detection rate:   {stats['detection_rate']:.1%}")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> np.ndarray:
        """
        Process a single frame and add overlays.

        Args:
            frame: Input frame
            frame_num: Frame number

        Returns:
            Frame with overlays
        """
        height, width = frame.shape[:2]
        output = frame.copy()

        # Detect person and get landmarks
        bbox, landmarks = self.person_detector.detect_person_and_landmarks(frame)

        # Define zone boundaries (always needed for drawing)
        zone_width = width // 3

        # Draw person bounding box
        if bbox:
            x, y, w, h, conf = bbox
            center_x = x + w / 2
            center_y = y + h / 2

            # Determine zone
            if center_x < zone_width:
                zone = 'LEFT'
                zone_color = (0, 0, 255)  # Red
            elif center_x < zone_width * 2:
                zone = 'CENTER'
                zone_color = (0, 255, 0)  # Green
            else:
                zone = 'RIGHT'
                zone_color = (255, 0, 0)  # Blue

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), zone_color, 3)

            # Draw center point
            cv2.circle(output, (int(center_x), int(center_y)), 5, zone_color, -1)

            # Calculate feet position (bottom of bbox, center horizontally)
            feet_x = int(center_x)
            feet_y = int(y + h)

            # Simple court check - assume court is in lower portion of frame
            on_court = feet_y > height * 0.5
            court_text = "ON-COURT" if on_court else "OFF-COURT"
            court_color = (0, 255, 0) if on_court else (0, 0, 255)

            # Draw feet point
            cv2.circle(output, (feet_x, feet_y), 3, court_color, -1)

            # Draw detailed info box at top-left of frame
            info_lines = [
                f"Person Box:",
                f"  pos=({x},{y}) size={w}x{h}",
                f"  center=({center_x:.0f},{center_y:.0f}) zone={zone}",
                f"  feet=({feet_x},{feet_y}) {court_text}",
                f"  conf={conf:.1%}"
            ]

            # Background for info box
            box_height = len(info_lines) * 22 + 10
            info_bg = np.zeros_like(output)
            cv2.rectangle(info_bg, (5, 5), (280, 5 + box_height), (0, 0, 0), -1)
            output = cv2.addWeighted(output, 0.7, info_bg, 0.5, 0)

            # Draw info text
            for i, line in enumerate(info_lines):
                text_color = zone_color if i == 0 else (255, 255, 255)
                if "OFF-COURT" in line:
                    text_color = (0, 0, 255)
                elif "ON-COURT" in line:
                    text_color = (0, 255, 0)
                cv2.putText(output, line, (10, 28 + i * 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Draw zone boundaries
        cv2.line(output, (zone_width, 0), (zone_width, height), (128, 128, 128), 1)
        cv2.line(output, (zone_width * 2, 0), (zone_width * 2, height), (128, 128, 128), 1)

        # Add zone labels
        cv2.putText(output, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, "CENTER", (zone_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, "RIGHT", (zone_width * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add frame info overlay
        info_bg = np.zeros_like(output)
        cv2.rectangle(info_bg, (10, height - 35), (200, height - 10), (0, 0, 0), -1)
        output = cv2.addWeighted(output, 0.5, info_bg, 0.3, 0)
        cv2.putText(output, f"Frame: {frame_num}", (10, height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw pose landmarks if enabled
        if self.show_pose and landmarks is not None:
            self._draw_landmarks(output, landmarks, width, height)

        return output

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        img_width: int,
        img_height: int
    ):
        """Draw pose landmarks and connections on frame."""
        # Draw connections first (so they're behind the points)
        for idx1, idx2, group in self.CONNECTIONS:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                lm1 = landmarks[idx1]
                lm2 = landmarks[idx2]

                # Only draw if both landmarks are visible
                if lm1[2] > 0.3 and lm2[2] > 0.3:
                    pt1 = (int(lm1[0] * img_width), int(lm1[1] * img_height))
                    pt2 = (int(lm2[0] * img_width), int(lm2[1] * img_height))
                    color = self.COLORS.get(group, self.COLORS['generic'])
                    cv2.line(frame, pt1, pt2, color, 2)

        # Draw landmark points
        for i, landmark in enumerate(landmarks):
            if landmark[2] > 0.3:  # Only draw visible landmarks
                pt = (int(landmark[0] * img_width), int(landmark[1] * img_height))
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Pose Overlay Processor - MediaPipe Only Version'
    )
    parser.add_argument(
        'input_video',
        help='Path to input video file'
    )
    parser.add_argument(
        'output_video',
        nargs='?',
        default=None,
        help='Path to output video file (optional)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Maximum number of frames to process (0 = all)'
    )
    parser.add_argument(
        '--bbox-only',
        action='store_true',
        help='Only show bounding boxes, hide pose skeleton'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_video):
        logger.error(f"Input video not found: {args.input_video}")
        sys.exit(1)

    # Generate output path if not provided
    if args.output_video is None:
        base, ext = os.path.splitext(args.input_video)
        suffix = "_BoxOnly" if args.bbox_only else "_PoseOverlay"
        args.output_video = f"{base}{suffix}_MediaPipe.mp4"

    logger.info("")
    logger.info("=" * 60)
    logger.info("POSE OVERLAY PROCESSOR - MediaPipe Only")
    logger.info("=" * 60)
    logger.info(f"Input:  {args.input_video}")
    logger.info(f"Output: {args.output_video}")
    logger.info(f"Mode:   {'BBox Only' if args.bbox_only else 'Pose + BBox'}")
    logger.info("")

    # Create processor
    processor = PoseOverlayProcessor(show_pose=not args.bbox_only)

    # Process video
    processor.process_video(
        args.input_video,
        args.output_video,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
