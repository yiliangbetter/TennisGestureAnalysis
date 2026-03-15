"""
Simple test to verify the Tennis Gesture Analysis system
"""
import os
import cv2
import numpy as np
from enhanced_gesture_analyzer import EnhancedTennisGestureAnalyzer, create_enhanced_sample_database

def create_test_video(filename, num_frames=30):
    """Create a simple test video for demonstration"""
    # Create a video with simulated movement
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))

    for i in range(num_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw a simple moving object to simulate a person
        center_x = int(width/2 + 100*np.sin(i*0.3))  # Oscillating horizontally
        center_y = int(height/2 + 50*np.cos(i*0.2))  # Oscillating vertically

        # Draw stick figure parts (simplified)
        # Head
        cv2.circle(frame, (center_x, center_y-80), 20, (255, 255, 255), -1)

        # Body
        cv2.line(frame, (center_x, center_y-60), (center_x, center_y), (255, 255, 255), 3)

        # Arms
        arm_end_x = int(center_x - 50*np.sin(i*0.3 + 0.5))
        arm_end_y = int(center_y-30 - 30*np.cos(i*0.3 + 0.5))
        cv2.line(frame, (center_x, center_y-40), (arm_end_x, arm_end_y), (255, 255, 255), 3)

        # Legs
        cv2.line(frame, (center_x, center_y), (center_x-20, center_y+40), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y), (center_x+20, center_y+40), (255, 255, 255), 3)

        out.write(frame)

    out.release()
    print(f"Created test video: {filename}")


def test_system():
    """Test the entire system"""
    print("Testing Tennis Gesture Analysis System...")

    # Create a test video
    test_video_path = "test_tennis_video.mp4"
    create_test_video(test_video_path, num_frames=50)

    # Initialize analyzer
    analyzer = EnhancedTennisGestureAnalyzer()

    # Create sample database
    print("Creating sample database...")
    create_enhanced_sample_database(analyzer)
    print(f"Database contains {len(analyzer.gesture_database)} samples")

    # Test finding best match (this would work with real video)
    print(f"\nTesting with video: {test_video_path}")
    result = analyzer.find_best_match(test_video_path)

    print("\nAnalysis Results:")
    if result['best_match']:
        print(f"Best Match: {result['best_match']}")
        print(f"Similarity Score: {result['similarity_score']:.2%}")
        print(f"Difference reports: {len(result['differences'])}")
        print(f"Recommendations: {len(result['recommendations'])}")

        print("\nSample Recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    else:
        print("No match found in the database.")

    # Clean up test video
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
        print(f"\nCleaned up test video: {test_video_path}")

    print("\nSystem test completed successfully!")


if __name__ == "__main__":
    test_system()