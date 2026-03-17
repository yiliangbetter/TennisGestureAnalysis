#!/usr/bin/env python3
"""
Debug script to visualize zone detection and verify the active player is correctly identified.
"""

import cv2
import numpy as np
import sys
from pose_overlay_processor import PersonDetector

def analyze_zone_detections(video_path: str, max_frames: int = 100):
    """Analyze which zones are being detected over time."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {total_frames} frames")
    print(f"Zone boundaries: LEFT=0-{width//3}, CENTER={width//3}-{2*width//3}, RIGHT={2*width//3}-{width}")
    print()

    detector = PersonDetector()

    zone_counts = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0, 'NONE': 0}
    zone_confidences = {'LEFT': [], 'CENTER': [], 'RIGHT': []}

    frame_count = 0
    recent_detections = []  # Store last 20 detections for analysis

    while True:
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
            elif center_x < zone_width * 2:
                zone = 'CENTER'
            else:
                zone = 'RIGHT'

            zone_counts[zone] += 1
            zone_confidences[zone].append(conf)

            recent_detections.append({
                'frame': frame_count,
                'zone': zone,
                'center_x': center_x,
                'confidence': conf,
                'bbox': (x, y, w, h)
            })

            # Keep only last 50 detections
            if len(recent_detections) > 50:
                recent_detections.pop(0)
        else:
            zone_counts['NONE'] += 1

        frame_count += 1

        if frame_count >= max_frames:
            break

    cap.release()

    # Print statistics
    print("=" * 60)
    print("ZONE DETECTION STATISTICS")
    print("=" * 60)
    print(f"Total frames analyzed: {frame_count}")
    print()

    for zone in ['LEFT', 'CENTER', 'RIGHT']:
        count = zone_counts[zone]
        pct = count / frame_count * 100
        avg_conf = np.mean(zone_confidences[zone]) if zone_confidences[zone] else 0
        print(f"{zone:8s}: {count:4d} frames ({pct:5.1f}%), avg confidence: {avg_conf:.2f}")

    print(f"NONE     : {zone_counts['NONE']:4d} frames ({zone_counts['NONE']/frame_count*100:.1f}%)")
    print()

    # Analyze recent detections (last 50 frames with detections)
    if recent_detections:
        print("=" * 60)
        print("RECENT DETECTIONS (last 50)")
        print("=" * 60)

        center_count = sum(1 for d in recent_detections if d['zone'] == 'CENTER')
        left_count = sum(1 for d in recent_detections if d['zone'] == 'LEFT')
        right_count = sum(1 for d in recent_detections if d['zone'] == 'RIGHT')

        total = len(recent_detections)
        print(f"CENTER: {center_count}/{total} ({center_count/total*100:.1f}%)")
        print(f"LEFT:   {left_count}/{total} ({left_count/total*100:.1f}%)")
        print(f"RIGHT:  {right_count}/{total} ({right_count/total*100:.1f}%)")
        print()

        # Check for stability (same zone over consecutive frames)
        if len(recent_detections) > 1:
            stable_transitions = 0
            total_transitions = 0
            for i in range(1, len(recent_detections)):
                if recent_detections[i]['frame'] == recent_detections[i-1]['frame'] + 1:
                    total_transitions += 1
                    if recent_detections[i]['zone'] == recent_detections[i-1]['zone']:
                        stable_transitions += 1

            if total_transitions > 0:
                stability = stable_transitions / total_transitions * 100
                print(f"Detection stability: {stability:.1f}% ({stable_transitions}/{total_transitions} same zone)")

    print()
    print("=" * 60)
    print("SAMPLE DETECTIONS (first 10)")
    print("=" * 60)

    # Re-run to get detailed output for first 10 frames
    cap = cv2.VideoCapture(video_path)
    detector2 = PersonDetector()

    for frame_num in range(min(10, max_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detector2.detect_person(frame)

        if bbox:
            x, y, w, h, conf = bbox
            center_x = x + w / 2
            center_y = y + h / 2
            zone_width = width // 3

            if center_x < zone_width:
                zone = 'LEFT'
            elif center_x < zone_width * 2:
                zone = 'CENTER'
            else:
                zone = 'RIGHT'

            print(f"Frame {frame_num:3d}: zone={zone:6s} bbox=({x:3d},{y:3d},{w:3d},{h:3d}) center=({center_x:.0f},{center_y:.0f}) conf={conf:.2f}")
        else:
            print(f"Frame {frame_num:3d}: NO DETECTION")

    cap.release()

    # Summary assessment
    print()
    print("=" * 60)
    print("ASSESSMENT")
    print("=" * 60)

    total_detected = zone_counts['LEFT'] + zone_counts['CENTER'] + zone_counts['RIGHT']
    if total_detected > 0:
        center_pct = zone_counts['CENTER'] / total_detected * 100
        left_pct = zone_counts['LEFT'] / total_detected * 100

        if center_pct >= 90:
            print("EXCELLENT: Center detection rate is very high!")
        elif center_pct >= 75:
            print("GOOD: Center detection is dominant")
        elif center_pct >= 60:
            print("MODERATE: Center detection is majority but side detections occur")
        else:
            print("POOR: Center detection needs improvement")

        print(f"  - CENTER: {center_pct:.1f}%")
        print(f"  - LEFT:   {left_pct:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_zone_detection.py <video_file> [max_frames]")
        sys.exit(1)

    video_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    analyze_zone_detections(video_path, max_frames)
