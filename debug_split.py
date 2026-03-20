#!/usr/bin/env python3
"""
Debug the split_wide_bbox function to understand why it's not selecting correct candidates.
"""

import cv2
import numpy as np
import sys
from pose_overlay_processor import PersonDetector

def debug_split_logic(video_path: str, target_frame: int = 350):
    """Debug the split logic on a specific frame."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    # Get to target frame
    for i in range(target_frame):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("Could not read target frame")
        return

    height, width = frame.shape[:2]
    print(f"Frame {target_frame}: {width}x{height}")
    print(f"Zone boundaries: LEFT=0-{width//3}, CENTER={width//3}-{2*width//3}, RIGHT={2*width//3}-{width}")
    print()

    # Manually run detection steps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Read previous frame for differencing
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    # Frame differencing
    frame_diff = cv2.absdiff(prev_gray, gray)
    _, diff_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)

    # Optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, flow_mask = cv2.threshold(mag_norm, 35, 255, cv2.THRESH_BINARY)

    # Combine
    combined = cv2.bitwise_or(diff_mask, flow_mask)

    # Accumulate (simulate)
    motion_accumulator = combined.astype(np.float32)
    accumulated_motion = np.clip(motion_accumulator, 0, 255).astype(np.uint8)
    _, accumulated_mask = cv2.threshold(accumulated_motion, 80, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    merged_mask = cv2.morphologyEx(accumulated_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Also check raw contours (before merging)
    raw_contours, _ = cv2.findContours(accumulated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(raw_contours)} raw contours (before merging)")
    print(f"Found {len(contours)} merged contours (after morphing)")

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w / 2
        zone = 'LEFT' if center_x < width // 3 else ('CENTER' if center_x < 2 * width // 3 else 'RIGHT')

        # Calculate aspect ratio
        aspect_ratio = h / max(w, 1)

        # Check max area filter
        max_area = width * height * 0.50
        skip_max_area = area > max_area

        # Check min bottom y filter
        min_bottom_y = int(height * 0.50)
        skip_bottom_y = (y + h) < min_bottom_y

        # Check aspect ratio filter
        skip_aspect = aspect_ratio < 0.9 or aspect_ratio > 4.0

        print(f"\nContour {i}:")
        print(f"  bbox: ({x}, {y}, {w}, {h})")
        print(f"  center_x: {center_x:.1f} ({zone} zone)")
        print(f"  area: {area}")
        print(f"  aspect_ratio: {aspect_ratio:.2f}")
        print(f"  width_ratio: {w/width:.2f} ({w/width*100:.1f}% of frame)")
        print(f"  bottom_y: {y+h} (min={min_bottom_y})")
        print(f"  FILTERS: skip_max_area={skip_max_area}, skip_bottom_y={skip_bottom_y}, skip_aspect={skip_aspect}")

        # Check if this would trigger split logic
        if w > width * 0.35:
            print(f"  -> Would trigger SPLIT (width > {width*0.35:.0f})")

            # Simulate split
            detector = PersonDetector()
            candidates = detector._split_wide_bbox(x, y, w, h, mag_norm, width, height, area)

            print(f"  -> Generated {len(candidates)} candidates:")
            for j, (cx, cy, cw, ch, ms) in enumerate(candidates):
                cand_center = cx + cw / 2
                cand_zone = 'LEFT' if cand_center < width // 3 else ('CENTER' if cand_center < 2 * width // 3 else 'RIGHT')
                print(f"     Candidate {j}: ({cx}, {cy}, {cw}, {h}) center={cand_center:.1f} ({cand_zone}) motion={ms:.3f}")
        else:
            print(f"  -> Would NOT trigger split (width {w} <= {width*0.35:.0f})")

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_split.py <video_file> [frame_number]")
        sys.exit(1)

    video_path = sys.argv[1]
    target_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 350

    debug_split_logic(video_path, target_frame)
