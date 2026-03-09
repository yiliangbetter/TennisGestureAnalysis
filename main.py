import cv2
import numpy as np
import argparse
import sys
import os
from enhanced_gesture_analyzer import EnhancedTennisGestureAnalyzer, create_enhanced_sample_database
from typing import Dict


def main():
    parser = argparse.ArgumentParser(description='Analyze tennis gestures and compare to professional players')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Path to save output video with analysis', default=None)
    parser.add_argument('--save_db', help='Path to save gesture database', default='gesture_database.pkl')
    parser.add_argument('--load_db', help='Path to load existing gesture database', default=None)

    args = parser.parse_args()

    # Validate input video exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video '{args.input_video}' does not exist.")
        sys.exit(1)

    # Initialize analyzer
    analyzer = TennisGestureAnalyzer()

    # Load or create database
    if args.load_db and os.path.exists(args.load_db):
        print(f"Loading gesture database from {args.load_db}")
        analyzer.load_database(args.load_db)
    else:
        print("Creating sample gesture database...")
        create_sample_database(analyzer)
        analyzer.save_database(args.save_db)
        print(f"Sample database saved to {args.save_db}")

    print(f"Database contains {len(analyzer.gesture_database)} gesture samples")

    # Analyze the input video
    print(f"\nAnalyzing input video: {args.input_video}")
    result = analyzer.find_best_match(args.input_video)

    # Display results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    if result['best_match']:
        print(f"Best Match: {result['best_match']}")
        print(f"Similarity Score: {result['similarity_score']:.2%}")

        print("\nKey Differences:")
        for i, diff in enumerate(result['differences'][:5]):  # Show first 5 differences
            print(f"  Frame {diff['frame_index']}: Landmark deviation = {np.mean(diff['landmark_deviation']):.3f}")

        print(f"\nRecommendations ({len(result['recommendations'])} suggestions):")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print("No match found in the database.")

    # Process video for visualization (if output path provided)
    if args.output:
        print(f"\nGenerating analysis video: {args.output}")
        generate_analysis_video(args.input_video, result, args.output, analyzer)

    print("\nAnalysis complete!")


def generate_analysis_video(input_path: str, analysis_result: Dict, output_path: str, analyzer: TennisGestureAnalyzer):
    """
    Generate an output video with analysis overlaid
    """
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to overlay analysis
        processed_frame = process_frame_for_analysis(frame, analysis_result, frame_count, analyzer)

        # Write the frame
        out.write(processed_frame)
        frame_count += 1

        # Show progress
        if frame_count % max(1, total_frames // 10) == 0:
            progress = frame_count / total_frames * 100
            print(f"Processing: {progress:.1f}%")

    # Release everything
    cap.release()
    out.release()


def process_frame_for_analysis(frame, analysis_result: Dict, frame_num: int, analyzer: TennisGestureAnalyzer):
    """
    Process a single frame to overlay analysis information
    """
    overlay = frame.copy()

    # Draw tennis court lines (simple representation)
    cv2.line(overlay, (width//2, 0), (width//2, height), (255, 255, 255), 2)  # Center line
    cv2.rectangle(overlay, (50, 50), (width-50, height-50), (255, 255, 255), 2)  # Outer boundary

    # Add analysis information to frame
    info_y = 30
    color = (0, 255, 0)  # Green text

    if analysis_result['best_match']:
        cv2.putText(overlay, f"Best Match: {analysis_result['best_match']}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        info_y += 30

        cv2.putText(overlay, f"Similarity: {analysis_result['similarity_score']:.2%}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        info_y += 30

        # Show a recommendation if available
        if analysis_result['recommendations']:
            cv2.putText(overlay, f"Tip: {analysis_result['recommendations'][0]}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Draw keypoints if available (simplified)
    # In a real implementation, we'd draw the actual detected pose

    # Blend original frame with overlay
    alpha = 0.7
    output_frame = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)

    return output_frame


if __name__ == "__main__":
    main()