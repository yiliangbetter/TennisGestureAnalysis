"""
Demo script for Tennis Gesture Analysis
"""

import os
import subprocess
import sys
from enhanced_gesture_analyzer import EnhancedTennisGestureAnalyzer, create_enhanced_sample_database


def run_demo():
    print("🎾 Tennis Gesture Analysis Demo 🎾")
    print("=" * 50)

    # Initialize analyzer
    analyzer = TennisGestureAnalyzer()

    # Create sample database
    print("Creating sample gesture database...")
    create_sample_database(analyzer)
    print(f"Database created with {len(analyzer.gesture_database)} samples")

    # Print available gestures
    print("\nAvailable professional gestures in database:")
    for i, name in enumerate(analyzer.gesture_database.keys(), 1):
        print(f"  {i}. {name}")

    print("\n" + "=" * 50)
    print("To analyze your own tennis video:")
    print("1. Prepare a video of yourself playing tennis")
    print("2. Run: python main.py <your_video.mp4>")
    print("3. Optionally add output option: python main.py <your_video.mp4> --output analysis.mp4")
    print("\nTo add new professional gestures to the database:")
    print("python add_gesture.py <professional_video.mp4> '<Player Name> - <Stroke Type>'")
    print("=" * 50)

    # Show how to use the system programmatically
    print("\nProgrammatic usage example:")
    print("# Initialize analyzer")
    print("analyzer = TennisGestureAnalyzer()")
    print("# Load or create database")
    print("create_sample_database(analyzer)")
    print("# Analyze a video")
    print("# result = analyzer.find_best_match('your_video.mp4')")
    print("# print(f'Best match: {result[\"best_match\"]}')")
    print("# print(f'Similarity: {result[\"similarity_score\"]:.2%}')")


if __name__ == "__main__":
    run_demo()