#!/usr/bin/env python3
"""
Debug script to identify specific frames where wrong zone is detected.
"""

import cv2
import numpy as np
import sys
from pose_overlay_processor import PersonDetector

def find_problem_frames(video_path: str):
    """Find and save frames where LEFT zone is detected instead of CENTER."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {total_frames} frames")

    detector = PersonDetector()

    left_frames = []
    center_frames = []

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detector.detect_person(frame)

        if bbox:
            x, y, w, h, conf = bbox
            center_x = x + w / 2
            zone_width = width // 3

            if center_x < zone_width:
                zone = 'LEFT'
                left_frames.append((frame_num, bbox, frame.copy()))
            elif center_x < zone_width * 2:
                zone = 'CENTER'
                center_frames.append((frame_num, bbox, frame.copy()))
            else:
                zone = 'RIGHT'

    cap.release()

    print(f"\nLEFT zone frames: {len(left_frames)}")
    print(f"CENTER zone frames: {len(center_frames)}")

    # Show some sample LEFT frames
    if left_frames:
        print("\n" + "=" * 60)
        print("SAMPLE LEFT ZONE FRAMES (first 10)")
        print("=" * 60)
        for frame_num, bbox, _ in left_frames[:10]:
            x, y, w, h, conf = bbox
            print(f"Frame {frame_num}: bbox=({x},{y},{w},{h}) conf={conf:.2f}")

    # Show transition points (where detection switches from CENTER to LEFT)
    print("\n" + "=" * 60)
    print("TRANSITION POINTS (CENTER -> LEFT)")
    print("=" * 60)

    all_detections = []
    cap = cv2.VideoCapture(video_path)
    detector2 = PersonDetector()

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        bbox = detector2.detect_person(frame)
        if bbox:
            x, y, w, h, conf = bbox
            center_x = x + w / 2
            zone_width = width // 3
            if center_x < zone_width:
                zone = 'LEFT'
            elif center_x < zone_width * 2:
                zone = 'CENTER'
            else:
                zone = 'RIGHT'
            all_detections.append((frame_num, zone, bbox))

    cap.release()

    # Find transitions
    transitions = []
    for i in range(1, len(all_detections)):
        prev_zone = all_detections[i-1][1]
        curr_zone = all_detections[i][1]
        if prev_zone != curr_zone:
            transitions.append((all_detections[i-1], all_detections[i]))

    print(f"Total zone transitions: {len(transitions)}")
    for prev_det, curr_det in transitions[:15]:
        print(f"  Frame {prev_det[0]} ({prev_det[1]}) -> Frame {curr_det[0]} ({curr_det[1]})")

    # Analyze the last portion of the video
    print("\n" + "=" * 60)
    print("LAST 100 FRAMES ANALYSIS")
    print("=" * 60)

    last_100 = all_detections[-100:] if len(all_detections) >= 100 else all_detections

    zone_counts = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0}
    for _, zone, _ in last_100:
        zone_counts[zone] += 1

    print(f"CENTER: {zone_counts['CENTER']} ({zone_counts['CENTER']/len(last_100)*100:.1f}%)")
    print(f"LEFT:   {zone_counts['LEFT']} ({zone_counts['LEFT']/len(last_100)*100:.1f}%)")
    print(f"RIGHT:  {zone_counts['RIGHT']} ({zone_counts['RIGHT']/len(last_100)*100:.1f}%)")

    # Check if player is moving to left side
    print("\nSample bboxes from last 20 frames:")
    for _, zone, bbox in all_detections[-20:]:
        x, y, w, h, conf = bbox
        center_x = x + w/2
        print(f"  {zone:6s}: center_x={center_x:.0f}, bbox=({x},{y},{w},{h})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_transitions.py <video_file>")
        sys.exit(1)

    find_problem_frames(sys.argv[1])
