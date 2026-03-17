#!/usr/bin/env python3
"""
Analyze motion distribution across zones to understand where the active player is.
"""

import cv2
import numpy as np
import sys

def analyze_motion_distribution(video_path: str, target_frames: list = None):
    """Analyze where motion is happening in the video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    zone_width = width // 3

    print(f"Video: {width}x{height}, {total_frames} frames")
    print(f"Zone boundaries: LEFT=0-{zone_width}, CENTER={zone_width}-{zone_width*2}, RIGHT={zone_width*2}-{width}")
    print()

    if target_frames is None:
        target_frames = list(range(total_frames))

    prev_gray = None
    frame_count = 0

    zone_motion_totals = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0}
    frame_motion_by_zone = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count not in target_frames:
            frame_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray
            frame_count += 1
            continue

        # Frame differencing
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, diff_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Count motion pixels per zone
        left_motion = np.count_nonzero(diff_mask[:, :zone_width])
        center_motion = np.count_nonzero(diff_mask[:, zone_width:zone_width*2])
        right_motion = np.count_nonzero(diff_mask[:, zone_width*2:])

        zone_motion_totals['LEFT'] += left_motion
        zone_motion_totals['CENTER'] += center_motion
        zone_motion_totals['RIGHT'] += right_motion

        total_motion = left_motion + center_motion + right_motion
        max_zone = max([('LEFT', left_motion), ('CENTER', center_motion), ('RIGHT', right_motion)], key=lambda x: x[1])

        frame_motion_by_zone.append({
            'frame': frame_count,
            'left': left_motion,
            'center': center_motion,
            'right': right_motion,
            'total': total_motion,
            'max_zone': max_zone[0]
        })

        prev_gray = gray
        frame_count += 1

    cap.release()

    # Print results
    print("=" * 60)
    print("MOTION DISTRIBUTION ACROSS ZONES")
    print("=" * 60)
    print(f"LEFT zone:   {zone_motion_totals['LEFT']:8d} motion pixels ({zone_motion_totals['LEFT']/sum(zone_motion_totals.values())*100:.1f}%)")
    print(f"CENTER zone: {zone_motion_totals['CENTER']:8d} motion pixels ({zone_motion_totals['CENTER']/sum(zone_motion_totals.values())*100:.1f}%)")
    print(f"RIGHT zone:  {zone_motion_totals['RIGHT']:8d} motion pixels ({zone_motion_totals['RIGHT']/sum(zone_motion_totals.values())*100:.1f}%)")
    print()

    # Count frames where each zone has most motion
    zone_max_counts = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0}
    for fm in frame_motion_by_zone:
        zone_max_counts[fm['max_zone']] += 1

    print("=" * 60)
    print("FRAMES WHERE EACH ZONE HAS MOST MOTION")
    print("=" * 60)
    total_analyzed = len(frame_motion_by_zone)
    for zone in ['LEFT', 'CENTER', 'RIGHT']:
        count = zone_max_counts[zone]
        pct = count / total_analyzed * 100
        print(f"{zone:6s}: {count:4d} frames ({pct:5.1f}%)")

    print()
    print("=" * 60)
    print("SAMPLE FRAMES")
    print("=" * 60)

    # Show first 20 frames
    for fm in frame_motion_by_zone[:20]:
        total = fm['total']
        left_pct = fm['left']/total*100 if total > 0 else 0
        center_pct = fm['center']/total*100 if total > 0 else 0
        right_pct = fm['right']/total*100 if total > 0 else 0
        print(f"Frame {fm['frame']:3d}: L={left_pct:5.1f}% C={center_pct:5.1f}% R={right_pct:5.1f}% max={fm['max_zone']}")

    # Show last 20 frames
    print("\nLast 20 frames:")
    for fm in frame_motion_by_zone[-20:]:
        total = fm['total']
        left_pct = fm['left']/total*100 if total > 0 else 0
        center_pct = fm['center']/total*100 if total > 0 else 0
        right_pct = fm['right']/total*100 if total > 0 else 0
        print(f"Frame {fm['frame']:3d}: L={left_pct:5.1f}% C={center_pct:5.1f}% R={right_pct:5.1f}% max={fm['max_zone']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_motion.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]

    # Analyze all frames
    analyze_motion_distribution(video_path)
