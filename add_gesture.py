"""
Utility script to add new tennis gestures to the database
"""

import cv2
import numpy as np
import pickle
import sys
import os
from gesture_analyzer_simple import TennisGestureAnalyzer, GestureFeature


def extract_gesture_from_video(video_path: str, gesture_name: str):
    """
    Extract gesture data from a video and add it to the database
    """
    # Initialize analyzer
    analyzer = TennisGestureAnalyzer()

    # Load existing database if it exists
    db_path = 'gesture_database.pkl'
    if os.path.exists(db_path):
        analyzer.load_database(db_path)
        print(f"Loaded existing database with {len(analyzer.gesture_database)} entries")

    # Extract features from the video
    print(f"Extracting features from {video_path}...")
    features = analyzer.extract_features_from_video(video_path)

    if not features:
        print(f"No features could be extracted from {video_path}")
        return False

    print(f"Extracted {len(features)} frames of gesture data")

    # Add to database
    analyzer.add_to_database(gesture_name, features)
    print(f"Added '{gesture_name}' to the database")

    # Save updated database
    analyzer.save_database(db_path)
    print(f"Database updated and saved to {db_path}")

    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python add_gesture.py <video_path> <gesture_name>")
        print("Example: python add_gesture.py federer_forehand.mp4 'Roger Federer - Forehand'")
        sys.exit(1)

    video_path = sys.argv[1]
    gesture_name = sys.argv[2]

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        sys.exit(1)

    success = extract_gesture_from_video(video_path, gesture_name)

    if success:
        print(f"Successfully added '{gesture_name}' to the gesture database!")
    else:
        print("Failed to add gesture to database.")


if __name__ == "__main__":
    main()