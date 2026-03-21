#!/usr/bin/env python3
"""
Pose Overlay Processor for Tennis Gesture Analysis

Processes tennis videos using MediaPipe Pose Landmarker for person detection
and pose landmark extraction. Overlays bounding box and skeleton on each frame.

Usage:
    python pose_overlay_processor.py <input_video> [output_video] [--bbox-only]

Example:
    python pose_overlay_processor.py SampleInputVideos.mp4 output.mp4
    python pose_overlay_processor.py SampleInputVideos.mp4 output.mp4 --bbox-only
"""

import cv2
import numpy as np
import argparse
import sys
import os
import logging
from datetime import datetime

# MediaPipe imports
try:
    import mediapipe as mp
    _vision = mp.tasks.vision
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = _vision.PoseLandmarker
    PoseLandmarkerOptions = _vision.PoseLandmarkerOptions
    RunningMode = _vision.RunningMode
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    BaseOptions = PoseLandmarker = PoseLandmarkerOptions = RunningMode = None


def setup_logging(log_dir: str = "logs"):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pose_processor_{timestamp}.log")

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


class PoseDetector:
    """MediaPipe-based person detector and landmark extractor."""

    def __init__(self, model_path: str = "pose_landmarker_heavy.task"):
        self.landmarker = None
        self.frames = 0
        self.detections = 0

        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe not available")
            return

        # Find model file
        for path in [model_path,
                     os.path.join(os.path.dirname(__file__), model_path),
                     os.path.join(os.getcwd(), model_path)]:
            if os.path.exists(path):
                model_path = path
                break
        else:
            logger.error(f"Model not found: {model_path}")
            return

        try:
            base_options = BaseOptions(model_asset_path=model_path)
            options = PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            logger.info(f"PoseLandmarker initialized: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")

    def detect(self, frame: np.ndarray):
        """
        Detect person and extract landmarks.

        Returns:
            Tuple of (bbox, landmarks) or (None, None)
            bbox: (x, y, w, h, confidence)
            landmarks: (33, 3) array of [x, y, visibility]
        """
        self.frames += 1

        if self.landmarker is None:
            return None, None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None, None

        self.detections += 1
        landmarks = result.pose_landmarks[0]

        # Convert to numpy array
        lm_array = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks], dtype=np.float32)

        # Compute bbox from landmarks
        bbox = self._landmarks_to_bbox(lm_array, w, h)

        return bbox, lm_array

    def _landmarks_to_bbox(self, landmarks, img_w, img_h, padding=0.1):
        """Compute bounding box from visible landmarks."""
        visible = landmarks[landmarks[:, 2] > 0.3]

        if len(visible) < 5:
            return None

        x_coords = visible[:, 0] * img_w
        y_coords = visible[:, 1] * img_h
        confidence = np.mean(visible[:, 2])

        # Add padding
        x_range = np.ptp(x_coords)
        y_range = np.ptp(y_coords)
        x = max(0, np.min(x_coords) - padding * x_range)
        y = max(0, np.min(y_coords) - padding * y_range)
        w = min(img_w, np.max(x_coords) + padding * x_range) - x
        h = min(img_h, np.max(y_coords) + padding * y_range) - y

        if w < 50 or h < 50:
            return None

        return (int(x), int(y), int(w), int(h), confidence)


class PoseOverlayProcessor:
    """Process video and overlay pose detection results."""

    # Body part colors (BGR)
    COLORS = {
        'nose': (0, 0, 255), 'eyes': (0, 128, 255), 'ears': (0, 128, 255),
        'shoulders': (255, 0, 0), 'elbows': (255, 0, 0), 'wrists': (255, 0, 255),
        'hips': (0, 255, 0), 'knees': (0, 255, 0), 'ankles': (128, 0, 255),
        'torso': (128, 128, 0), 'default': (255, 255, 255)
    }

    # Landmark connections (idx1, idx2, group)
    CONNECTIONS = [
        (0, 1, 'nose'), (0, 2, 'nose'), (1, 3, 'ears'), (2, 4, 'ears'),
        (11, 12, 'torso'), (11, 23, 'torso'), (12, 24, 'torso'), (23, 24, 'torso'),
        (11, 13, 'shoulders'), (13, 15, 'elbows'),
        (12, 14, 'shoulders'), (14, 16, 'elbows'),
        (23, 25, 'hips'), (25, 27, 'knees'),
        (24, 26, 'hips'), (26, 28, 'knees'),
    ]

    def __init__(self, show_pose: bool = True):
        self.detector = PoseDetector()
        self.show_pose = show_pose

    def process_video(self, input_path: str, output_path: str, max_frames: int = 0):
        """Process video and save with overlays."""
        logger.info(f"Opening: {input_path}")
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            logger.error(f"Failed to open: {input_path}")
            sys.exit(1)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {w}x{h}, {fps:.1f} FPS, {total} frames")

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not out.isOpened():
            logger.error("Failed to create output")
            sys.exit(1)

        logger.info(f"Output: {output_path}")
        logger.info("Processing...")

        detected = 0
        positions = []

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            overlay = self.process_frame(frame, i)
            out.write(overlay)

            bbox, _ = self.detector.detect(frame)
            if bbox:
                detected += 1
                positions.append((bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2))

            if (i + 1) % 50 == 0 or i + 1 == total:
                logger.info(f"Progress: {100*(i+1)/total:.1f}% ({i+1}/{total})")

            if max_frames > 0 and i + 1 >= max_frames:
                break

        cap.release()
        out.release()

        # Stats
        logger.info("")
        logger.info("=" * 50)
        logger.info("STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Frames processed: {self.detector.frames}")
        logger.info(f"Frames detected:  {self.detector.detections}")
        logger.info(f"Detection rate:   {100*self.detector.detections/max(self.detector.frames,1):.1f}%")

        if positions:
            pos = np.array(positions)
            var = np.mean(np.var(pos, axis=0))
            logger.info(f"Position variance: {var:.2f}")

        logger.info(f"Output: {output_path}")

    def process_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """Process single frame and add overlays."""
        h, w = frame.shape[:2]
        output = frame.copy()

        bbox, landmarks = self.detector.detect(frame)
        zone_w = w // 3

        # Draw zone boundaries
        cv2.line(output, (zone_w, 0), (zone_w, h), (128, 128, 128), 1)
        cv2.line(output, (zone_w*2, 0), (zone_w*2, h), (128, 128, 128), 1)
        cv2.putText(output, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(output, "CENTER", (zone_w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(output, "RIGHT", (zone_w*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if bbox:
            x, y, bw, bh, conf = bbox
            cx, cy = x + bw/2, y + bh/2

            # Zone color
            if cx < zone_w:
                zone, color = 'LEFT', (0, 0, 255)
            elif cx < zone_w * 2:
                zone, color = 'CENTER', (0, 255, 0)
            else:
                zone, color = 'RIGHT', (255, 0, 0)

            # Bounding box
            cv2.rectangle(output, (x, y), (x+bw, y+bh), color, 3)
            cv2.circle(output, (int(cx), int(cy)), 5, color, -1)

            # Court check
            feet_y = y + bh
            on_court = feet_y > h * 0.5
            court_text = "ON-COURT" if on_court else "OFF-COURT"
            court_color = (0, 255, 0) if on_court else (0, 0, 255)
            cv2.circle(output, (int(cx), feet_y), 3, court_color, -1)

            # Info box
            lines = [
                f"Box: ({x},{y}) {bw}x{bh}",
                f"Center: ({cx:.0f},{cy:.0f}) Zone: {zone}",
                f"Feet: {court_text}",
                f"Conf: {conf:.1%}"
            ]
            bg_h = len(lines) * 22 + 10
            bg = np.zeros_like(output)
            cv2.rectangle(bg, (5, 5), (260, 5+bg_h), (0,0,0), -1)
            output = cv2.addWeighted(output, 0.7, bg, 0.5, 0)

            for i, line in enumerate(lines):
                cv2.putText(output, line, (10, 28+i*22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Frame number
        bg = np.zeros_like(output)
        cv2.rectangle(bg, (10, h-35), (200, h-10), (0,0,0), -1)
        output = cv2.addWeighted(output, 0.5, bg, 0.3, 0)
        cv2.putText(output, f"Frame: {frame_num}", (10, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Draw pose skeleton
        if self.show_pose and landmarks is not None:
            self._draw_landmarks(output, landmarks, w, h)

        return output

    def _draw_landmarks(self, frame, landmarks, img_w, img_h):
        """Draw pose landmarks."""
        # Draw connections
        for i1, i2, group in self.CONNECTIONS:
            if landmarks[i1, 2] > 0.3 and landmarks[i2, 2] > 0.3:
                pt1 = (int(landmarks[i1, 0] * img_w), int(landmarks[i1, 1] * img_h))
                pt2 = (int(landmarks[i2, 0] * img_w), int(landmarks[i2, 1] * img_h))
                color = self.COLORS.get(group, self.COLORS['default'])
                cv2.line(frame, pt1, pt2, color, 2)

        # Draw points
        for lm in landmarks:
            if lm[2] > 0.3:
                pt = (int(lm[0] * img_w), int(lm[1] * img_h))
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)


def main():
    parser = argparse.ArgumentParser(description='Pose Overlay Processor')
    parser.add_argument('input_video', help='Input video file')
    parser.add_argument('output_video', nargs='?', default=None, help='Output video file')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames (0=all)')
    parser.add_argument('--bbox-only', action='store_true', help='Hide pose skeleton')

    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        logger.error(f"Not found: {args.input_video}")
        sys.exit(1)

    if args.output_video is None:
        base, _ = os.path.splitext(args.input_video)
        suffix = "_BoxOnly" if args.bbox_only else "_PoseOverlay"
        args.output_video = f"{base}{suffix}.mp4"

    logger.info("")
    logger.info("=" * 50)
    logger.info("POSE OVERLAY PROCESSOR")
    logger.info("=" * 50)
    logger.info(f"Input:  {args.input_video}")
    logger.info(f"Output: {args.output_video}")
    logger.info(f"Mode:   {'Box Only' if args.bbox_only else 'Pose + Box'}")
    logger.info("")

    processor = PoseOverlayProcessor(show_pose=not args.bbox_only)
    processor.process_video(args.input_video, args.output_video, args.max_frames)


if __name__ == "__main__":
    main()
