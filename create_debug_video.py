#!/usr/bin/env python3
"""
Create a debug video showing detection boxes and zone information.
"""

import cv2
import numpy as np
import sys
from pose_overlay_processor import PersonDetector

def create_debug_video(input_path: str, output_path: str, max_frames: int = 200):
    """Create a debug video showing detection boxes."""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = PersonDetector()

    zone_counts = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0, 'NONE': 0}
    zone_width = width // 3

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw zone boundaries
        cv2.line(frame, (zone_width, 0), (zone_width, height), (255, 255, 255), 1)
        cv2.line(frame, (zone_width * 2, 0), (zone_width * 2, height), (255, 255, 255), 1)

        # Add zone labels
        cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "CENTER", (zone_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "RIGHT", (zone_width * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Detect person
        bbox = detector.detect_person(frame)

        if bbox:
            x, y, w, h, conf = bbox
            center_x = x + w / 2

            if center_x < zone_width:
                zone = 'LEFT'
                color = (0, 0, 255)  # Red for LEFT (bad)
            elif center_x < zone_width * 2:
                zone = 'CENTER'
                color = (0, 255, 0)  # Green for CENTER (good)
            else:
                zone = 'RIGHT'
                color = (255, 0, 0)  # Blue for RIGHT

            zone_counts[zone] += 1

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            # Draw center point
            cv2.circle(frame, (int(center_x), int(y + h / 2)), 5, color, -1)

            # Draw info
            info_text = f"{zone}: {conf:.1%}"
            cv2.putText(frame, info_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw width info
            width_text = f"w={w}"
            cv2.putText(frame, width_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            zone_counts['NONE'] += 1

        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add running stats
        total = sum(zone_counts.values())
        if total > 0:
            stats = f"C:{zone_counts['CENTER']}/{total} L:{zone_counts['LEFT']}/{total}"
            cv2.putText(frame, stats, (width - 200, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1

        if frame_count >= max_frames:
            break

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()

    print(f"\nDebug video saved to: {output_path}")
    print(f"Total frames: {frame_count}")
    print(f"CENTER: {zone_counts['CENTER']} ({zone_counts['CENTER']/frame_count*100:.1f}%)")
    print(f"LEFT:   {zone_counts['LEFT']} ({zone_counts['LEFT']/frame_count*100:.1f}%)")
    print(f"RIGHT:  {zone_counts['RIGHT']} ({zone_counts['RIGHT']/frame_count*100:.1f}%)")
    print(f"NONE:   {zone_counts['NONE']} ({zone_counts['NONE']/frame_count*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_debug_video.py <input_video> [output_video] [max_frames]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "debug_output.mp4"
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    create_debug_video(input_path, output_path, max_frames)
