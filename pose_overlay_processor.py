#!/usr/bin/env python3
"""
Pose Overlay Processor for Tennis Gesture Analysis

This script processes tennis videos, detects pose landmarks, and overlays
visualizations on each frame for analysis and debugging.

Usage:
    python pose_overlay_processor.py <input_video> [output_video]

Example:
    python pose_overlay_processor.py SampleInputVideos.mp4 SampleInputVideosPoseOverlay.mp4
"""

import cv2
import numpy as np
import argparse
import sys
import os
from typing import Optional, List, Dict, Tuple
from enhanced_gesture_analyzer import EnhancedTennisGestureAnalyzer, MEDIAPIPE_LEGACY


class PersonDetector:
    """
    Detect person in frame using motion cues.
    Optimized for tennis video - detects the ACTIVE player only.

    Key insight: Tennis players create motion throughout their body during
    swings. Static observers create minimal motion. We detect the largest
    coherent motion region that has person-like proportions.
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_bbox = None
        self.frames_without_detection = 0

    def detect_person(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Detect the ACTIVE tennis player using frame differencing.

        Returns:
            (x, y, w, h, confidence) or None if no person detected
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Need at least 2 frames for differencing
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Frame differencing with LOW threshold to capture all motion
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, diff_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Optical flow for additional motion signal
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, flow_mask = cv2.threshold(mag_norm, 40, 255, cv2.THRESH_BINARY)

        # Combine masks - union of both methods
        combined = cv2.bitwise_or(diff_mask, flow_mask)

        # Morphological cleanup - close gaps to connect motion fragments
        # Use larger kernel and more iterations to connect fragmented motion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find ALL contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            # Minimum area: 0.5% of frame (was 2%, too restrictive)
            min_area = width * height * 0.003

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w/2, y + h/2

            # Person-like aspect ratio (was 0.8-2.2, now 0.5-2.5 to include swinging)
            aspect_ratio = h / max(w, 1)
            if aspect_ratio < 0.4 or aspect_ratio > 3.0:
                continue

            # Width filter - but more lenient (was 35%, now 50%)
            if w > width * 0.6:
                continue

            # Define zones for bias calculation
            zone_width = width // 3

            # Determine zone
            if center_x < zone_width:
                contour_zone = 'LEFT'
            elif center_x < zone_width * 2:
                contour_zone = 'CENTER'
            else:
                contour_zone = 'RIGHT'

            # Calculate motion intensity in this region from flow
            roi_flow = mag_norm[y:y+h, x:x+w]
            avg_motion = np.mean(roi_flow) if roi_flow.size > 0 else 0

            # Motion density
            motion_pixels = cv2.countNonZero(roi_flow)
            motion_density = motion_pixels / roi_flow.size if roi_flow.size > 0 else 0

            # Base motion score
            motion_score = (avg_motion / 255) * 0.5 + motion_density * 0.5

            # Center preference (active player tends to be center-court)
            center_bonus = 1.0 - abs(center_x - width/2) / (width/2)

            # STRONG zone bias - tennis player is almost always in CENTER zone
            # This prevents false positives on static observers on the sides
            zone_bias = 0.0
            if contour_zone == 'CENTER':
                zone_bias = 0.5  # 50% bonus for CENTER zone
            elif contour_zone == 'LEFT':
                zone_bias = -0.3  # 30% penalty for LEFT zone

            # Prefer bounding boxes that start in upper-middle (where players are)
            y_position_score = 1.0 if y < height * 0.7 else 0.5

            # Combined score
            score = area * motion_score * (1 + 0.5 * center_bonus + zone_bias) * y_position_score

            # Temporal consistency - prefer bbox near previous detection
            if self.prev_bbox:
                px, py, pw, ph, _ = self.prev_bbox[:5]
                prev_center_x = px + pw/2
                prev_center_y = py + ph/2
                dist = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
                if dist < 250:
                    score *= 1.3

            if score > best_score:
                best_score = score
                best_bbox = (x, y, w, h)

        # Return best detection
        if best_bbox:
            x, y, w, h = best_bbox
            self.frames_without_detection = 0
            confidence = min(1.0, best_score / (width * height * 0.05))
            self.prev_bbox = (x, y, w, h, confidence)
            self.prev_gray = gray
            return (x, y, w, h, confidence)

        # Fallback: if no motion detected, don't return stale bbox
        # Let the pose estimation work without a prior
        self.frames_without_detection += 1

        # Clear previous bbox after 5 frames without detection
        # This prevents stale landmarks from persisting
        if self.frames_without_detection > 5:
            self.prev_bbox = None

        self.prev_gray = gray
        return None


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

    # Landmark names for display
    LANDMARK_NAMES = {
        0: 'nose', 1: 'L_eye', 2: 'R_eye', 3: 'L_ear', 4: 'R_ear',
        11: 'L_sho', 12: 'R_sho', 13: 'L_elb', 14: 'R_elb',
        15: 'L_wri', 16: 'R_wri',
        23: 'L_hip', 24: 'R_hip', 25: 'L_kne', 26: 'R_kne',
        27: 'L_ank', 28: 'R_ank'
    }

    def __init__(self, show_connections: bool = True, show_labels: bool = True,
                 landmark_radius: int = 5, line_thickness: int = 2):
        """
        Initialize the Pose Overlay Processor.

        Args:
            show_connections: Whether to draw skeleton connections
            show_labels: Whether to show landmark labels
            landmark_radius: Radius of landmark points
            line_thickness: Thickness of connection lines
        """
        self.show_connections = show_connections
        self.show_labels = show_labels
        self.landmark_radius = landmark_radius
        self.line_thickness = line_thickness

        # Initialize person detector
        self.person_detector = PersonDetector()

        # Initialize gesture analyzer for pose detection
        self.analyzer = EnhancedTennisGestureAnalyzer()

    def process_video(self, input_path: str, output_path: Optional[str] = None,
                     show_progress: bool = True) -> List[Dict]:
        """
        Process a video file and overlay pose detections.

        Args:
            input_path: Path to input video file
            output_path: Path to output video (optional, if None just returns landmarks)
            show_progress: Whether to print progress messages

        Returns:
            List of dictionaries containing landmarks for each frame
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if show_progress:
            print(f"Input video: {input_path}")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {fps:.1f}")
            print(f"Total frames: {total_frames}")

        # Create output writer if output path provided
        out = None
        if output_path:
            # Try H.264 codec first (best compatibility), fall back to mp4v
            fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')
            fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')

            # Test if H.264 is available
            test_writer = cv2.VideoWriter("/tmp/test_codec.mp4", fourcc_h264, fps, (width, height))
            if test_writer.isOpened():
                test_writer.release()
                fourcc = fourcc_h264
                codec_name = 'H.264 (avc1)'
            else:
                fourcc = fourcc_mp4v
                codec_name = 'MPEG-4 (mp4v)'
                print(f"Note: H.264 not available, using {codec_name}")

            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output codec: {codec_name}")

        # Store landmarks for each frame
        all_landmarks = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect person in frame
            person_bbox = self.person_detector.detect_person(frame)

            # Extract landmarks from frame with person location
            landmarks = self.analyzer.extract_landmarks_from_frame(
                frame, person_bbox=person_bbox
            )

            # Store landmarks
            all_landmarks.append({
                'frame': frame_count,
                'landmarks': landmarks.copy() if landmarks is not None else None,
                'detected': landmarks is not None,
                'person_bbox': person_bbox
            })

            # Create overlay
            overlay = self.overlay_landmarks(frame, landmarks, frame_count, person_bbox)

            # Write to output
            if out is not None:
                out.write(overlay)

            frame_count += 1

            # Show progress
            if show_progress and frame_count % max(1, total_frames // 10) == 0:
                progress = frame_count / total_frames * 100
                print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames})")

        # Cleanup
        cap.release()
        if out is not None:
            out.release()

        if show_progress:
            print(f"\nProcessed {frame_count} frames")
            if output_path:
                print(f"Output saved to: {output_path}")

        return all_landmarks

    def overlay_landmarks(self, frame: np.ndarray, landmarks: Optional[np.ndarray],
                         frame_num: int = 0, person_bbox: Optional[Tuple] = None) -> np.ndarray:
        """
        Overlay pose landmarks on a frame.

        Args:
            frame: Input frame (BGR)
            landmarks: 33x2 array of normalized landmarks (or None)
            frame_num: Frame number for display
            person_bbox: Optional (x, y, w, h, confidence) tuple

        Returns:
            Frame with overlaid pose visualization
        """
        output = frame.copy()
        height, width = output.shape[:2]

        # Draw person detection bounding box
        if person_bbox:
            x, y, bw, bh, conf = person_bbox
            cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.putText(output, f"Person: {conf:.1%}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Add frame info overlay
        info_bg = np.zeros_like(output)
        cv2.rectangle(info_bg, (0, 0), (width, 40), (0, 0, 0), -1)
        output = cv2.addWeighted(output, 0.7, info_bg, 0.3, 0)

        cv2.putText(output, f"Frame: {frame_num}", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add detection status
        status_color = (0, 255, 0) if landmarks is not None else (0, 0, 255)
        status = "DETECTED" if landmarks is not None else "NOT DETECTED"
        cv2.putText(output, f"Pose: {status}", (width - 200, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if landmarks is None:
            return output

        # Draw connections (skeleton)
        if self.show_connections:
            for idx1, idx2, color_name in self.CONNECTIONS:
                if idx1 < len(landmarks) and idx2 < len(landmarks):
                    pt1 = self._normalize_to_pixel(landmarks[idx1], width, height)
                    pt2 = self._normalize_to_pixel(landmarks[idx2], width, height)

                    if pt1 and pt2:
                        color = self.COLORS.get(color_name, self.COLORS['generic'])
                        cv2.line(output, pt1, pt2, color, self.line_thickness)

        # Draw landmark points
        for i, landmark in enumerate(landmarks):
            pt = self._normalize_to_pixel(landmark, width, height)
            if pt:
                color = self.COLORS.get(self.LANDMARK_NAMES.get(i, 'generic'),
                                       self.COLORS['generic'])
                cv2.circle(output, pt, self.landmark_radius, color, -1)

                # Draw label if enabled
                if self.show_labels and i in self.LANDMARK_NAMES:
                    label_pos = (pt[0] + 8, pt[1] - 5)
                    cv2.putText(output, self.LANDMARK_NAMES[i], label_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw center of mass
        if len(landmarks) > 24:
            torso_center = np.mean(landmarks[[11, 12, 23, 24]], axis=0)
            center_pt = self._normalize_to_pixel(torso_center, width, height)
            if center_pt:
                cv2.circle(output, center_pt, 8, (0, 255, 255), -1)

        return output

    def _normalize_to_pixel(self, landmark: np.ndarray, width: int,
                           height: int) -> Optional[tuple]:
        """Convert normalized landmark coordinates to pixel coordinates."""
        if landmark is None or len(landmark) < 2:
            return None

        x = int(np.clip(landmark[0], 0, 1) * width)
        y = int(np.clip(landmark[1], 0, 1) * height)

        # Validate coordinates are within frame
        if 0 <= x < width and 0 <= y < height:
            return (x, y)
        return None

    def generate_statistics(self, landmarks_data: List[Dict]) -> Dict:
        """
        Generate statistics from landmark data.

        Args:
            landmarks_data: List of landmark dictionaries from process_video

        Returns:
            Dictionary containing pose detection statistics
        """
        total_frames = len(landmarks_data)
        detected_frames = sum(1 for data in landmarks_data if data['detected'])

        # Calculate detection rate
        detection_rate = detected_frames / total_frames if total_frames > 0 else 0

        # Analyze landmark confidence and positions
        avg_positions = None
        position_variance = None

        if detected_frames > 0:
            all_positions = []
            for data in landmarks_data:
                if data['landmarks'] is not None:
                    all_positions.append(data['landmarks'])

            if all_positions:
                stacked = np.stack(all_positions)
                avg_positions = np.mean(stacked, axis=0)
                position_variance = np.var(stacked, axis=0)

        return {
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'detection_rate': detection_rate,
            'avg_positions': avg_positions,
            'position_variance': position_variance
        }

    def close(self):
        """Release resources."""
        self.analyzer.close()


def main():
    """Main entry point for pose overlay processing."""
    parser = argparse.ArgumentParser(
        description='Process tennis videos and overlay pose detections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pose_overlay_processor.py input.mp4 output.mp4
  python pose_overlay_processor.py SampleInputVideos.mp4
  python pose_overlay_processor.py video.mp4 overlay.mp4 --no-labels
        """
    )

    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', nargs='?', default=None,
                       help='Path to output video (default: <input>_PoseOverlay.mp4)')
    parser.add_argument('--no-connections', action='store_true',
                       help='Disable skeleton connections display')
    parser.add_argument('--no-labels', action='store_true',
                       help='Disable landmark labels display')
    parser.add_argument('--radius', type=int, default=6,
                       help='Landmark point radius (default: 6)')
    parser.add_argument('--thickness', type=int, default=2,
                       help='Connection line thickness (default: 2)')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_video):
        print(f"Error: Input video '{args.input_video}' not found.")
        sys.exit(1)

    # Generate default output path if not provided
    if args.output_video is None:
        base, ext = os.path.splitext(args.input_video)
        args.output_video = f"{base}_PoseOverlay{ext}"

    print("=" * 60)
    print("POSE OVERLAY PROCESSOR")
    print("=" * 60)
    print(f"MediaPipe Legacy API: {'Available' if MEDIAPIPE_LEGACY else 'Not available'}")
    print(f"Input:  {args.input_video}")
    print(f"Output: {args.output_video}")
    print()

    # Create processor
    processor = PoseOverlayProcessor(
        show_connections=not args.no_connections,
        show_labels=not args.no_labels,
        landmark_radius=args.radius,
        line_thickness=args.thickness
    )

    try:
        # Process video
        landmarks_data = processor.process_video(args.input_video, args.output_video)

        # Generate and display statistics
        stats = processor.generate_statistics(landmarks_data)

        print("\n" + "=" * 60)
        print("PROCESSING STATISTICS")
        print("=" * 60)
        print(f"Total frames:       {stats['total_frames']}")
        print(f"Detected frames:    {stats['detected_frames']}")
        print(f"Detection rate:     {stats['detection_rate']:.1%}")

        if stats['position_variance'] is not None:
            avg_variance = np.mean(stats['position_variance'])
            print(f"Avg position var:   {avg_variance:.6f}")

        print("\nProcessing complete!")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        processor.close()


if __name__ == "__main__":
    main()
