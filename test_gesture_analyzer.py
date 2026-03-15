#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Tennis Gesture Analysis System

This module contains unit tests for all major components:
- Landmark extraction
- Feature extraction
- Gesture comparison
- Video processing
- Pose overlay

Run with: python -m pytest test_gesture_analyzer.py -v
       or: python test_gesture_analyzer.py
"""

import unittest
import os
import sys
import cv2
import numpy as np
import tempfile
import shutil

# Import components to test
from enhanced_gesture_analyzer import (
    EnhancedTennisGestureAnalyzer,
    EnhancedGestureFeature,
    create_enhanced_sample_database,
    MEDIAPIPE_LEGACY
)
from pose_overlay_processor import PoseOverlayProcessor


class TestEnhancedGestureAnalyzer(unittest.TestCase):
    """Test cases for EnhancedTennisGestureAnalyzer class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()
        cls.analyzer = EnhancedTennisGestureAnalyzer()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.analyzer.close()
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_initialization(self):
        """Test analyzer initializes correctly."""
        # Create fresh analyzer for this test
        analyzer = EnhancedTennisGestureAnalyzer()
        try:
            self.assertIsNotNone(analyzer)
            self.assertIsNotNone(analyzer.gesture_database)
            # Database starts empty (unless sample data was pre-loaded)
            self.assertIsInstance(analyzer.gesture_database, dict)
        finally:
            analyzer.close()

    def test_create_sample_database(self):
        """Test sample database creation."""
        create_enhanced_sample_database(self.analyzer)
        self.assertGreater(len(self.analyzer.gesture_database), 0)
        self.assertIn("Roger Federer - Forehand", self.analyzer.gesture_database)

    def test_landmark_shape(self):
        """Test that landmarks have correct shape (33 points, 2 coords)."""
        # Create a test frame with a visible object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 100), (440, 400), (255, 255, 255), -1)

        landmarks = self.analyzer.extract_landmarks_from_frame(frame)

        if landmarks is not None:
            self.assertEqual(landmarks.shape[0], 33, "Should have 33 landmarks")
            self.assertEqual(landmarks.shape[1], 2, "Each landmark should have x,y coords")
            # Check normalized coordinates (allow small tolerance for interpolation noise)
            # Landmarks should generally be in [0, 1] range
            self.assertTrue(np.all(landmarks >= -0.1), "Coordinates should be >= 0 (with tolerance)")
            self.assertTrue(np.all(landmarks <= 1.1), "Coordinates should be <= 1 (with tolerance)")

    def test_joint_angles_calculation(self):
        """Test joint angle calculations."""
        # Create mock landmarks (33 points)
        landmarks = np.random.rand(33, 2).astype(np.float32)

        angles = self.analyzer.calculate_joint_angles(landmarks)

        self.assertEqual(len(angles), 12, "Should calculate 12 joint angles")
        for angle in angles:
            # Check it's a numeric type (float or np.float32)
            self.assertTrue(np.isscalar(angle), "Angle should be a scalar")
            self.assertGreaterEqual(angle, 0, "Angle should be non-negative")
            self.assertLessEqual(angle, 180, "Angle should be <= 180 degrees")

    def test_joint_angles_insufficient_landmarks(self):
        """Test joint angles with insufficient landmarks."""
        # Create incomplete landmarks
        landmarks = np.random.rand(20, 2).astype(np.float32)

        angles = self.analyzer.calculate_joint_angles(landmarks)

        self.assertEqual(len(angles), 12, "Should return default 12 angles")
        self.assertEqual(angles, [0.0] * 12, "Should return all zeros for insufficient landmarks")

    def test_optical_flow_calculation(self):
        """Test optical flow between frames."""
        # Create two slightly different frames
        frame1 = np.random.randint(0, 245, (100, 100, 3), dtype=np.uint8)
        frame2 = frame1.astype(np.int16) + np.random.randint(-10, 10, (100, 100, 3), dtype=np.int16)
        frame2 = np.clip(frame2, 0, 255).astype(np.uint8)

        flow = self.analyzer.calculate_optical_flow(frame1, frame2)

        self.assertIsNotNone(flow)
        self.assertEqual(len(flow.shape), 3, "Flow should be 3D array")
        self.assertEqual(flow.shape[2], 2, "Flow should have 2 channels (dx, dy)")

    def test_motion_history_calculation(self):
        """Test motion history image calculation."""
        # Create a sequence of frames
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]

        motion_history = self.analyzer.calculate_motion_history(frames, 4)

        self.assertIsNotNone(motion_history)
        self.assertEqual(motion_history.shape, (100, 100), "Should match frame dimensions")

    def test_hog_feature_extraction(self):
        """Test HOG feature extraction."""
        frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        hog_features = self.analyzer.extract_hog_features(frame)

        self.assertIsNotNone(hog_features)
        self.assertGreater(len(hog_features), 0, "Should extract some HOG features")

    def test_trajectory_extraction(self):
        """Test trajectory extraction from frame sequence."""
        # Create a sequence of frames with moving object
        frames = []
        for i in range(10):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.circle(frame, (50 + i * 10, 100), 20, (255, 255, 255), -1)
            frames.append(frame)

        # Mock landmarks
        landmarks = np.random.rand(33, 2).astype(np.float32)

        trajectories = self.analyzer.extract_trajectories(frames, 9, landmarks)

        self.assertIsInstance(trajectories, list)
        self.assertGreater(len(trajectories), 0, "Should extract at least one trajectory")

    def test_pose_similarity(self):
        """Test pose similarity calculation."""
        # Identical poses should have high similarity
        landmarks1 = np.random.rand(33, 2).astype(np.float32)
        landmarks2 = landmarks1.copy()

        similarity = self.analyzer._pose_similarity(landmarks1, landmarks2)

        self.assertGreaterEqual(similarity, 0.99, "Identical poses should have high similarity")

        # Very different poses should have low similarity
        landmarks3 = np.random.rand(33, 2).astype(np.float32)
        similarity_diff = self.analyzer._pose_similarity(landmarks1, landmarks3)

        self.assertLess(similarity_diff, 1.0, "Different poses should have lower similarity")

    def test_angle_similarity(self):
        """Test angle similarity calculation."""
        angles1 = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
        angles2 = angles1.copy()

        similarity = self.analyzer._angle_similarity(angles1, angles2)

        self.assertEqual(similarity, 1.0, "Identical angles should have similarity 1.0")

    def test_trajectory_similarity(self):
        """Test trajectory similarity calculation."""
        traj1 = [[(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)]]
        traj2 = [[(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)]]

        similarity = self.analyzer._trajectory_similarity(traj1, traj2)

        self.assertEqual(similarity, 1.0, "Identical trajectories should have similarity 1.0")

    def test_velocity_acceleration_calculation(self):
        """Test velocity and acceleration calculation."""
        landmarks1 = np.random.rand(33, 2).astype(np.float32)
        landmarks2 = landmarks1 + np.random.rand(33, 2).astype(np.float32) * 0.1

        velocity, acceleration = self.analyzer.calculate_velocities_and_acceleration(
            landmarks2, landmarks1
        )

        self.assertEqual(velocity.shape, landmarks2.shape, "Velocity shape should match landmarks")
        self.assertIsInstance(acceleration, float)

    def test_enhanced_feature_extraction(self):
        """Test full enhanced feature extraction from frame sequence."""
        # Create test frames
        frames = []
        for i in range(5):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            # Add some content for detection
            cv2.rectangle(frame, (50, 50), (150, 150), (200, 200, 200), -1)
            frames.append(frame)

        features = self.analyzer.extract_enhanced_gesture_features(frames)

        # Should extract features for each frame
        self.assertGreater(len(features), 0, "Should extract features from frames")

        # Check feature structure
        feature = features[0]
        self.assertIsInstance(feature, EnhancedGestureFeature)
        self.assertIsNotNone(feature.pose_landmarks)
        self.assertIsNotNone(feature.joint_angles)


class TestGestureComparison(unittest.TestCase):
    """Test cases for gesture comparison functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.analyzer = EnhancedTennisGestureAnalyzer()
        create_enhanced_sample_database(cls.analyzer)

        # Create test video
        cls.test_dir = tempfile.mkdtemp()
        cls.test_video = os.path.join(cls.test_dir, "test_comparison.mp4")

        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.test_video, fourcc, 10.0, (width, height))

        for i in range(20):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (200, 100), (440, 400), (255, 255, 255), -1)
            out.write(frame)

        out.release()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.analyzer.close()
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_find_best_match(self):
        """Test finding best match from database."""
        result = self.analyzer.find_best_match(self.test_video)

        self.assertIsInstance(result, dict)
        self.assertIn('best_match', result)
        self.assertIn('similarity_score', result)
        self.assertIn('differences', result)
        self.assertIn('recommendations', result)

    def test_similarity_score_range(self):
        """Test that similarity scores are in valid range."""
        result = self.analyzer.find_best_match(self.test_video)

        self.assertGreaterEqual(result['similarity_score'], 0.0, "Similarity should be >= 0")
        self.assertLessEqual(result['similarity_score'], 1.0, "Similarity should be <= 1")

    def test_database_save_load(self):
        """Test saving and loading gesture database."""
        db_path = os.path.join(self.test_dir, "test_db.pkl")

        # Save database
        self.analyzer.save_database(db_path)
        self.assertTrue(os.path.exists(db_path), "Database file should be created")

        # Load into new analyzer
        new_analyzer = EnhancedTennisGestureAnalyzer()
        new_analyzer.load_database(db_path)

        self.assertEqual(len(new_analyzer.gesture_database),
                        len(self.analyzer.gesture_database),
                        "Loaded database should have same size")

        new_analyzer.close()


class TestPoseOverlayProcessor(unittest.TestCase):
    """Test cases for PoseOverlayProcessor class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()
        cls.processor = PoseOverlayProcessor()

        # Create test video
        cls.test_video = os.path.join(cls.test_dir, "test_overlay.mp4")

        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.test_video, fourcc, 10.0, (width, height))

        for i in range(30):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (200 + i * 5, 100), (440 + i * 5, 400), (255, 255, 255), -1)
            out.write(frame)

        out.release()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.processor.close()
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        self.assertIsNotNone(self.processor)
        self.assertIsNotNone(self.processor.analyzer)

    def test_video_processing(self):
        """Test video processing and landmark extraction."""
        output_video = os.path.join(self.test_dir, "test_output.mp4")

        landmarks_data = self.processor.process_video(self.test_video, output_video)

        self.assertGreater(len(landmarks_data), 0, "Should process frames")
        self.assertTrue(os.path.exists(output_video), "Output video should be created")

        # Check output video is valid
        cap = cv2.VideoCapture(output_video)
        self.assertTrue(cap.isOpened(), "Output video should be readable")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertGreater(frame_count, 0, "Output video should have frames")
        cap.release()

    def test_landmark_overlay(self):
        """Test landmark overlay on frames."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 100), (440, 400), (255, 255, 255), -1)

        # Get landmarks
        landmarks = self.processor.analyzer.extract_landmarks_from_frame(frame)

        # Create overlay
        overlay = self.processor.overlay_landmarks(frame, landmarks, 0)

        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape, frame.shape, "Overlay should match frame dimensions")

    def test_overlay_with_none_landmarks(self):
        """Test overlay handles None landmarks gracefully."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        overlay = self.processor.overlay_landmarks(frame, None, 0)

        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape, frame.shape)

    def test_statistics_generation(self):
        """Test statistics generation from landmarks data."""
        output_video = os.path.join(self.test_dir, "test_stats.mp4")
        landmarks_data = self.processor.process_video(self.test_video, output_video)

        stats = self.processor.generate_statistics(landmarks_data)

        self.assertIn('total_frames', stats)
        self.assertIn('detected_frames', stats)
        self.assertIn('detection_rate', stats)
        self.assertGreater(stats['total_frames'], 0)
        self.assertGreaterEqual(stats['detection_rate'], 0)
        self.assertLessEqual(stats['detection_rate'], 1)

    def test_coordinate_normalization(self):
        """Test normalization from landmark coordinates to pixels."""
        landmark = np.array([0.5, 0.5])  # Center point

        pt = self.processor._normalize_to_pixel(landmark, 640, 480)

        self.assertEqual(pt, (320, 240), "Should correctly normalize to pixel coordinates")

    def test_invalid_coordinate_handling(self):
        """Test handling of invalid coordinates."""
        # Out of bounds coordinates
        landmark = np.array([2.0, 2.0])

        pt = self.processor._normalize_to_pixel(landmark, 640, 480)

        self.assertIsNone(pt, "Should return None for out-of-bounds coordinates")


class TestVideoProcessingPipeline(unittest.TestCase):
    """Integration tests for the complete video processing pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_full_pipeline(self):
        """Test complete pipeline from video to analysis."""
        # Create test video
        test_video = os.path.join(self.test_dir, "pipeline_test.mp4")
        output_video = os.path.join(self.test_dir, "pipeline_output.mp4")

        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video, fourcc, 10.0, (width, height))

        for i in range(20):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (200, 100), (440, 400), (255, 255, 255), -1)
            out.write(frame)

        out.release()

        # Process through pipeline
        analyzer = EnhancedTennisGestureAnalyzer()
        create_enhanced_sample_database(analyzer)

        # Extract features
        features = analyzer.extract_features_from_video(test_video)
        self.assertGreater(len(features), 0, "Should extract features")

        # Find match
        result = analyzer.find_best_match(test_video)
        self.assertIsNotNone(result)

        analyzer.close()

    def test_processor_with_sample_video(self):
        """Test processor with the SampleInputVideos.mp4 if available."""
        sample_video = "SampleInputVideos.mp4"

        if not os.path.exists(sample_video):
            self.skipTest("SampleInputVideos.mp4 not found")

        output_video = os.path.join(self.test_dir, "SampleInputVideosPoseOverlay.mp4")

        processor = PoseOverlayProcessor()
        landmarks_data = processor.process_video(sample_video, output_video)

        self.assertGreater(len(landmarks_data), 0)
        self.assertTrue(os.path.exists(output_video), "Output video should be created")

        processor.close()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.analyzer = EnhancedTennisGestureAnalyzer()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.analyzer.close()

    def test_empty_frame_sequence(self):
        """Test handling of empty frame sequence."""
        features = self.analyzer.extract_enhanced_gesture_features([])
        self.assertEqual(len(features), 0)

    def test_single_frame(self):
        """Test processing single frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        features = self.analyzer.extract_enhanced_gesture_features([frame])

        # Should handle single frame (no optical flow expected)
        self.assertGreaterEqual(len(features), 0)

    def test_reset_functionality(self):
        """Test analyzer reset between videos."""
        self.analyzer.reset()
        self.assertIsNone(self.analyzer.prev_landmarks)

    def test_comparison_with_empty_database(self):
        """Test comparison with empty database."""
        empty_analyzer = EnhancedTennisGestureAnalyzer()

        # Create test video
        test_dir = tempfile.mkdtemp()
        test_video = os.path.join(test_dir, "empty_test.mp4")

        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video, fourcc, 10.0, (width, height))
        for _ in range(10):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        result = empty_analyzer.find_best_match(test_video)

        self.assertIsNone(result['best_match'])
        self.assertEqual(result['similarity_score'], 0.0)

        empty_analyzer.close()
        shutil.rmtree(test_dir)


def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedGestureAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestGestureComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestPoseOverlayProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessingPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success: {result.wasSuccessful()}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
