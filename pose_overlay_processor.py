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

    IMPROVED: Uses contour merging to connect fragmented motion from the
    active player, and multi-frame motion accumulation for robust detection.
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_bbox = None
        self.frames_without_detection = 0
        # Motion accumulation for multi-frame analysis
        self.motion_accumulator = None
        self.accumulation_count = 0
        self.max_accumulation_frames = 5

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

        # Frame differencing with adaptive threshold
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, diff_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)

        # Optical flow for additional motion signal
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, flow_mask = cv2.threshold(mag_norm, 35, 255, cv2.THRESH_BINARY)

        # Combine masks - union of both methods
        combined = cv2.bitwise_or(diff_mask, flow_mask)

        # Accumulate motion over multiple frames for more complete player silhouette
        if self.motion_accumulator is None:
            self.motion_accumulator = combined.astype(np.float32)
        else:
            # Add new motion with decay
            self.motion_accumulator = self.motion_accumulator * 0.7 + combined.astype(np.float32) * 0.3
        self.accumulation_count += 1

        # Use accumulated motion for contour detection
        accumulated_motion = np.clip(self.motion_accumulator, 0, 255).astype(np.uint8)
        _, accumulated_mask = cv2.threshold(accumulated_motion, 80, 255, cv2.THRESH_BINARY)

        # MORPHOLOGICAL MERGING: Use very aggressive closing to connect fragmented motion
        # This is key - tennis player motion is fragmented across arms, legs, torso
        # A large kernel connects these into a single "person" contour
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        merged_mask = cv2.morphologyEx(accumulated_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Also apply opening to remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Find contours on merged mask
        contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Also find contours on raw combined mask for comparison
        raw_contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = 0
        best_zone = None

        # Evaluate merged contours first (these represent connected motion regions)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Lower minimum area since we merged fragments
            min_area = width * height * 0.002

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w/2, y + h/2

            # Person-like aspect ratio (more lenient for merged contours)
            aspect_ratio = h / max(w, 1)
            if aspect_ratio < 0.3 or aspect_ratio > 4.0:
                continue

            # CRITICAL: Width filter - reject detections that are too wide
            # A single person should be at most ~35% of frame width
            # Wider detections likely include multiple people or background motion
            if w > width * 0.35:
                # Try to split wide detections - see if we can find person-sized regions
                # within this bounding box
                split_candidates = self._split_wide_bbox(
                    x, y, w, h, mag_norm, width, height, area
                )
                for candidate in split_candidates:
                    cx, cy, cw, ch, motion_score = candidate
                    # Evaluate the split candidate
                    cand_center_x = cx + cw / 2
                    cand_center_y = cy + ch / 2
                    cand_zone_width = width // 3

                    if cand_center_x < cand_zone_width:
                        cand_zone = 'LEFT'
                    elif cand_center_x < cand_zone_width * 2:
                        cand_zone = 'CENTER'
                    else:
                        cand_zone = 'RIGHT'

                    cand_center_bonus = 1.0 - abs(cand_center_x - width/2) / (width/2)
                    # No zone bias - track the player wherever they move
                    cand_zone_bias = 0.0

                    cand_y_score = 1.0 if cy < height * 0.75 else 0.5

                    cand_score = (area * 0.3 + motion_score * 1000) * (1 + cand_center_bonus + cand_zone_bias) * cand_y_score

                    if self.prev_bbox:
                        px, py, pw, ph, _ = self.prev_bbox[:5]
                        prev_center_x = px + pw/2
                        prev_center_y = py + ph/2
                        dist = np.sqrt((cand_center_x - prev_center_x)**2 + (cand_center_y - prev_center_y)**2)
                        if dist < 200:
                            cand_score *= 1.5

                    if cand_score > best_score:
                        best_score = cand_score
                        best_bbox = (cx, cy, cw, ch)
                        best_zone = cand_zone
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

            # Motion density - what fraction of the ROI has motion
            motion_pixels = cv2.countNonZero(roi_flow)
            motion_density = motion_pixels / roi_flow.size if roi_flow.size > 0 else 0

            # Base motion score
            motion_score = (avg_motion / 255) * 0.5 + motion_density * 0.5

            # Center preference (active player tends to be center-court)
            center_bonus = 1.0 - abs(center_x - width/2) / (width/2)

            # NO zone bias - track the player wherever they move on the court
            # Zone is only used for logging/analysis, not for scoring
            zone_bias = 0.0

            # Prefer bounding boxes that start in upper-middle (where players are)
            y_position_score = 1.0 if y < height * 0.75 else 0.5

            # Combined score
            score = (area * 0.3 + motion_score * 1000) * (1 + center_bonus + zone_bias) * y_position_score

            # Temporal consistency - STRONG preference for bbox near previous detection
            # Active player moves smoothly, static observers appear suddenly
            if self.prev_bbox:
                px, py, pw, ph, _ = self.prev_bbox[:5]
                prev_center_x = px + pw/2
                prev_center_y = py + ph/2
                dist = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
                if dist < 200:  # Tighter threshold
                    score *= 1.5  # Stronger bonus for continuity

            if score > best_score:
                best_score = score
                best_bbox = (x, y, w, h)
                best_zone = contour_zone

        # If no good merged contour found, fall back to evaluating raw contours
        # This catches cases where merging was too aggressive
        if best_bbox is None:
            # Group nearby raw contours into clusters
            clusters = self._cluster_contours(raw_contours, max_distance=50)

            for cluster in clusters:
                # Compute bounding box for cluster
                all_points = np.vstack([contour.reshape(-1, 2) for contour in cluster])
                x, y, w, h = cv2.boundingRect(all_points)
                area = sum(cv2.contourArea(c) for c in cluster)
                center_x, center_y = x + w/2, y + h/2

                min_area = width * height * 0.001  # Even lower for clusters

                if area < min_area:
                    continue

                aspect_ratio = h / max(w, 1)
                if aspect_ratio < 0.3 or aspect_ratio > 4.0:
                    continue

                if w > width * 0.7:
                    continue

                zone_width = width // 3
                if center_x < zone_width:
                    contour_zone = 'LEFT'
                elif center_x < zone_width * 2:
                    contour_zone = 'CENTER'
                else:
                    contour_zone = 'RIGHT'

                # Calculate motion intensity
                roi_flow = mag_norm[y:y+h, x:x+w]
                avg_motion = np.mean(roi_flow) if roi_flow.size > 0 else 0
                motion_pixels = cv2.countNonZero(roi_flow)
                motion_density = motion_pixels / roi_flow.size if roi_flow.size > 0 else 0
                motion_score = (avg_motion / 255) * 0.5 + motion_density * 0.5

                center_bonus = 1.0 - abs(center_x - width/2) / (width/2)

                # No zone bias - track the player wherever they move
                zone_bias = 0.0

                y_position_score = 1.0 if y < height * 0.75 else 0.5
                score = (area * 0.3 + motion_score * 1000) * (1 + center_bonus + zone_bias) * y_position_score

                if self.prev_bbox:
                    px, py, pw, ph, _ = self.prev_bbox[:5]
                    prev_center_x = px + pw/2
                    prev_center_y = py + ph/2
                    dist = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
                    if dist < 200:
                        score *= 1.5

                if score > best_score:
                    best_score = score
                    best_bbox = (x, y, w, h)
                    best_zone = contour_zone

        # Return best detection
        if best_bbox:
            x, y, w, h = best_bbox
            self.frames_without_detection = 0
            confidence = min(1.0, best_score / (width * height * 0.03))
            self.prev_bbox = (x, y, w, h, confidence)
            self.prev_gray = gray
            return (x, y, w, h, confidence)

        # Fallback: if no motion detected, don't return stale bbox
        self.frames_without_detection += 1

        # Clear previous bbox after 3 frames without detection (faster decay)
        if self.frames_without_detection > 3:
            self.prev_bbox = None
            # Reset motion accumulator
            self.motion_accumulator = None
            self.accumulation_count = 0

        self.prev_gray = gray
        return None

    def _cluster_contours(self, contours: List[np.ndarray],
                          max_distance: float = 50) -> List[List[np.ndarray]]:
        """
        Group nearby contours into clusters.

        Args:
            contours: List of contours to cluster
            max_distance: Maximum distance between contours to be in same cluster

        Returns:
            List of contour clusters
        """
        if not contours:
            return []

        # Filter out tiny contours first
        min_contour_area = 50  # pixels
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        if not valid_contours:
            return []

        # Simple clustering: assign each contour to nearest cluster
        clusters = [[valid_contours[0]]]

        for contour in valid_contours[1:]:
            bbox = cv2.boundingRect(contour)
            center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)

            best_cluster = None
            best_dist = float('inf')

            for i, cluster in enumerate(clusters):
                # Find distance to nearest contour in cluster
                for other in cluster:
                    other_bbox = cv2.boundingRect(other)
                    other_center = (other_bbox[0] + other_bbox[2]/2, other_bbox[1] + other_bbox[3]/2)
                    dist = np.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = i

            if best_dist <= max_distance:
                clusters[best_cluster].append(contour)
            else:
                clusters.append([contour])

        return clusters

    def _split_wide_bbox(self, x: int, y: int, w: int, h: int,
                         mag_norm: np.ndarray, img_w: int, img_h: int,
                         total_area: float) -> List[Tuple[int, int, int, int, float]]:
        """
        Split a wide bounding box into person-sized candidates.

        When a detection is too wide (>45% of frame), it likely includes multiple
        people or background motion. This method tries to find person-sized regions
        within the wide bbox.

        Args:
            x, y, w, h: Original bounding box
            mag_norm: Optical flow magnitude image
            img_w, img_h: Image dimensions
            total_area: Total contour area

        Returns:
            List of (x, y, w, h, motion_score) tuples for candidate regions
        """
        candidates = []

        # Split the wide bbox into 3 overlapping regions and find the best person-sized one
        # Person width should be roughly 15-25% of frame width
        target_person_width = int(img_w * 0.20)
        min_person_width = int(img_w * 0.12)
        max_person_width = int(img_w * 0.35)

        # Slide a window across the bbox to find person-sized regions with high motion
        step = target_person_width // 2

        for offset_x in range(0, w - target_person_width + 1, step):
            sub_x = x + offset_x
            sub_w = target_person_width

            # Calculate motion in this sub-region
            roi_flow = mag_norm[y:y+h, sub_x:sub_x+sub_w]
            if roi_flow.size == 0:
                continue

            avg_motion = np.mean(roi_flow)
            motion_pixels = cv2.countNonZero(roi_flow)
            motion_density = motion_pixels / roi_flow.size

            # Only consider regions with significant motion
            if motion_density < 0.05 or avg_motion < 20:
                continue

            # Check if this region has person-like properties
            aspect_ratio = h / sub_w
            if aspect_ratio < 0.8 or aspect_ratio > 3.5:
                continue

            motion_score = (avg_motion / 255) * 0.5 + motion_density * 0.5
            candidates.append((sub_x, y, sub_w, h, motion_score))

        # Also try to find the region with highest motion concentration
        # by dividing the bbox into left/center/right thirds
        third_w = w // 3
        for i in range(3):
            sub_x = x + i * third_w
            sub_w = third_w

            # Ensure minimum width
            if sub_w < min_person_width:
                sub_w = min_person_width

            roi_flow = mag_norm[y:y+h, sub_x:sub_x+sub_w]
            if roi_flow.size == 0:
                continue

            avg_motion = np.mean(roi_flow)
            motion_pixels = cv2.countNonZero(roi_flow)
            motion_density = motion_pixels / roi_flow.size

            if motion_density < 0.05 or avg_motion < 20:
                continue

            motion_score = (avg_motion / 255) * 0.5 + motion_density * 0.5
            candidates.append((sub_x, y, sub_w, h, motion_score))

        # Return candidates sorted by motion score (highest first)
        candidates.sort(key=lambda c: c[4], reverse=True)

        # Return top 2 candidates (in case there are two people)
        return candidates[:2]


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
