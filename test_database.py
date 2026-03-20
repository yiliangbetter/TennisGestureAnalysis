#!/usr/bin/env python3
"""
Unit tests for Tennis Gesture Analysis database module.

Tests cover:
- Database initialization and schema creation
- Player operations (add, get, list)
- Video operations (add, get, mark processed)
- Pose operations (add, get, query)
- Similarity search
- Gesture sequences
- Comparison results
"""

import unittest
import sqlite3
import numpy as np
import os
import tempfile
from pathlib import Path

from database_manager import TennisDatabase, PoseData


class TestTennisDatabase(unittest.TestCase):
    """Test cases for TennisDatabase class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temp database for each test
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_tennis.db")
        self.db = TennisDatabase(self.db_path)

    def tearDown(self):
        """Clean up after tests"""
        self.db.close()
        # Remove temp database file
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    # =========================================================================
    # Database Initialization Tests
    # =========================================================================

    def test_database_creation(self):
        """Test that database file is created"""
        self.assertTrue(os.path.exists(self.db_path))

    def test_schema_tables_exist(self):
        """Test that all required tables are created"""
        expected_tables = [
            'players', 'videos', 'extracted_poses',
            'pose_landmarks', 'joint_angles', 'gesture_sequences'
        ]

        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row['name'] for row in cursor.fetchall()]

        for table in expected_tables:
            self.assertIn(table, tables)

    def test_schema_indexes_exist(self):
        """Test that required indexes are created"""
        expected_indexes = [
            'idx_poses_video', 'idx_poses_player',
            'idx_poses_sample', 'idx_landmarks_pose', 'idx_angles_pose'
        ]

        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = [row['name'] for row in cursor.fetchall()]

        for index in expected_indexes:
            self.assertIn(index, indexes)

    # =========================================================================
    # Player Operations Tests
    # =========================================================================

    def test_add_player(self):
        """Test adding a player"""
        player_id = self.db.add_player(
            name="Novak Djokovic",
            country="SRB",
            is_professional=True
        )

        self.assertIsInstance(player_id, int)
        self.assertGreater(player_id, 0)

    def test_add_player_duplicate(self):
        """Test adding duplicate player returns existing ID"""
        id1 = self.db.add_player(name="Test Player", country="USA")
        id2 = self.db.add_player(name="Test Player", country="USA")

        self.assertEqual(id1, id2)

    def test_get_player(self):
        """Test getting player by ID"""
        player_id = self.db.add_player(
            name="Rafael Nadal",
            country="ESP",
            is_professional=True
        )

        player = self.db.get_player(player_id)

        self.assertIsNotNone(player)
        self.assertEqual(player['name'], "Rafael Nadal")
        self.assertEqual(player['country'], "ESP")
        self.assertTrue(player['is_professional'])

    def test_get_player_by_name(self):
        """Test getting player by name"""
        self.db.add_player(name="Roger Federer", country="SUI")

        player = self.db.get_player_by_name("Roger Federer")

        self.assertIsNotNone(player)
        self.assertEqual(player['name'], "Roger Federer")

    def test_get_nonexistent_player(self):
        """Test getting nonexistent player returns None"""
        player = self.db.get_player(9999)
        self.assertIsNone(player)

    def test_get_all_players(self):
        """Test getting all players"""
        self.db.add_player("Player 1", "USA")
        self.db.add_player("Player 2", "ESP")
        self.db.add_player("Player 3", "GER")

        players = self.db.get_all_players()

        self.assertEqual(len(players), 3)

    def test_get_professional_players_only(self):
        """Test filtering professional players"""
        self.db.add_player("Pro Player", "USA", is_professional=True)
        self.db.add_player("Amateur Player", "USA", is_professional=False)

        pros = self.db.get_all_players(professionals_only=True)

        self.assertEqual(len(pros), 1)
        self.assertEqual(pros[0]['name'], "Pro Player")

    # =========================================================================
    # Video Operations Tests
    # =========================================================================

    def test_add_video(self):
        """Test adding a video"""
        video_id = self.db.add_video(
            filename="test_video.mp4",
            file_path="/path/to/test_video.mp4",
            metadata={
                'duration': 10.5,
                'fps': 30,
                'width': 1920,
                'height': 1080,
                'total_frames': 315
            }
        )

        self.assertIsInstance(video_id, int)
        self.assertGreater(video_id, 0)

    def test_add_video_duplicate(self):
        """Test adding duplicate video returns existing ID"""
        id1 = self.db.add_video(
            filename="test.mp4",
            file_path="/path/to/test.mp4"
        )
        id2 = self.db.add_video(
            filename="test.mp4",
            file_path="/path/to/test.mp4"
        )

        self.assertEqual(id1, id2)

    def test_get_video(self):
        """Test getting video by ID"""
        video_id = self.db.add_video(
            filename="my_video.mp4",
            file_path="/videos/my_video.mp4",
            metadata={'fps': 60, 'width': 3840, 'height': 2160}
        )

        video = self.db.get_video(video_id)

        self.assertIsNotNone(video)
        self.assertEqual(video['filename'], "my_video.mp4")
        self.assertEqual(video['fps'], 60)

    def test_mark_video_processed(self):
        """Test marking video as processed"""
        video_id = self.db.add_video(
            filename="unprocessed.mp4",
            file_path="/path/to/unprocessed.mp4"
        )

        # Initially should be unprocessed
        video = self.db.get_video(video_id)
        self.assertFalse(video['processed'])

        # Mark as processed
        self.db.mark_video_processed(video_id)

        # Check updated status
        video = self.db.get_video(video_id)
        self.assertTrue(video['processed'])

    # =========================================================================
    # Pose Operations Tests
    # =========================================================================

    def _create_test_pose_data(self, video_id, player_id, frame_num=0):
        """Helper to create test pose data"""
        landmarks = np.random.rand(33, 2).astype(np.float32)

        return PoseData(
            id=None,
            video_id=video_id,
            frame_number=frame_num,
            player_id=player_id,
            landmarks=landmarks,
            joint_angles={
                'right_elbow_flexion': 90.0,
                'left_elbow_flexion': 85.0,
            },
            bbox=(0.1, 0.2, 0.3, 0.4),
            stroke_type="forehand",
            confidence=0.95,
            timestamp_ms=frame_num * 33.33
        )

    def test_add_pose(self):
        """Test adding a pose"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_data = self._create_test_pose_data(video_id, player_id)

        pose_id = self.db.add_pose(pose_data)

        self.assertIsInstance(pose_id, int)
        self.assertGreater(pose_id, 0)

    def test_add_pose_saves_landmarks(self):
        """Test that all 33 landmarks are saved"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_data = self._create_test_pose_data(video_id, player_id)

        pose_id = self.db.add_pose(pose_data)

        # Count landmarks
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM pose_landmarks WHERE pose_id = ?",
                (pose_id,)
            )
            count = cursor.fetchone()[0]

        self.assertEqual(count, 33)

    def test_add_pose_saves_angles(self):
        """Test that joint angles are saved"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_data = self._create_test_pose_data(video_id, player_id)

        pose_id = self.db.add_pose(pose_data)

        # Count angles
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM joint_angles WHERE pose_id = ?",
                (pose_id,)
            )
            count = cursor.fetchone()[0]

        self.assertEqual(count, 2)  # We added 2 angles

    def test_get_pose(self):
        """Test getting complete pose by ID"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        original_pose = self._create_test_pose_data(video_id, player_id)

        pose_id = self.db.add_pose(original_pose)
        retrieved_pose = self.db.get_pose(pose_id)

        self.assertIsNotNone(retrieved_pose)
        self.assertEqual(retrieved_pose.id, pose_id)
        self.assertEqual(retrieved_pose.video_id, video_id)
        self.assertEqual(retrieved_pose.player_id, player_id)
        self.assertEqual(retrieved_pose.landmarks.shape, (33, 2))
        self.assertIn('right_elbow_flexion', retrieved_pose.joint_angles)

    def test_get_pose_landmarks_correct(self):
        """Test that retrieved landmarks match original"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Create pose with specific landmarks
        specific_landmarks = np.linspace(0, 1, 66).reshape(33, 2).astype(np.float32)
        pose_data = self._create_test_pose_data(video_id, player_id)
        pose_data.landmarks = specific_landmarks

        pose_id = self.db.add_pose(pose_data)
        retrieved_pose = self.db.get_pose(pose_id)

        np.testing.assert_array_almost_equal(
            retrieved_pose.landmarks, specific_landmarks, decimal=5
        )

    def test_get_poses_by_video(self):
        """Test getting all poses from a video"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Add multiple poses
        for i in range(5):
            pose_data = self._create_test_pose_data(video_id, player_id, frame_num=i)
            self.db.add_pose(pose_data)

        poses = self.db.get_poses_by_video(video_id)

        self.assertEqual(len(poses), 5)

    def test_get_poses_by_player(self):
        """Test getting all poses for a player"""
        player1_id = self.db.add_player("Player 1")
        player2_id = self.db.add_player("Player 2")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Add poses for both players
        for i in range(3):
            pose_data = self._create_test_pose_data(video_id, player1_id, frame_num=i)
            self.db.add_pose(pose_data)

            pose_data = self._create_test_pose_data(video_id, player2_id, frame_num=i+10)
            self.db.add_pose(pose_data)

        player1_poses = self.db.get_poses_by_player(player1_id)
        player2_poses = self.db.get_poses_by_player(player2_id)

        self.assertEqual(len(player1_poses), 3)
        self.assertEqual(len(player2_poses), 3)

    def test_get_poses_by_player_with_stroke_filter(self):
        """Test filtering poses by stroke type"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Add forehand poses
        for i in range(3):
            pose_data = self._create_test_pose_data(video_id, player_id, frame_num=i)
            pose_data.stroke_type = "forehand"
            self.db.add_pose(pose_data)

        # Add backhand poses
        for i in range(2):
            pose_data = self._create_test_pose_data(video_id, player_id, frame_num=i+10)
            pose_data.stroke_type = "backhand"
            self.db.add_pose(pose_data)

        forehand_poses = self.db.get_poses_by_player(player_id, stroke_type="forehand")
        backhand_poses = self.db.get_poses_by_player(player_id, stroke_type="backhand")

        self.assertEqual(len(forehand_poses), 3)
        self.assertEqual(len(backhand_poses), 2)

    # =========================================================================
    # Sample Pose Tests
    # =========================================================================

    def test_get_sample_poses(self):
        """Test getting sample poses"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Add regular pose
        pose_data = self._create_test_pose_data(video_id, player_id, frame_num=0)
        self.db.add_pose(pose_data)

        # Add sample pose
        pose_data = self._create_test_pose_data(video_id, player_id, frame_num=10)
        pose_data.is_sample_pose = True
        self.db.add_pose(pose_data)

        sample_poses = self.db.get_sample_poses()

        self.assertEqual(len(sample_poses), 1)

    def test_get_sample_poses_filter_by_player(self):
        """Test filtering sample poses by player name"""
        player1_id = self.db.add_player("Player 1")
        player2_id = self.db.add_player("Player 2")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Add sample poses for both players
        pose_data = self._create_test_pose_data(video_id, player1_id)
        self.db.add_pose(pose_data)
        with self.db.connection() as conn:
            conn.execute("UPDATE extracted_poses SET is_sample_pose = 1")

        pose_data = self._create_test_pose_data(video_id, player2_id)
        self.db.add_pose(pose_data)
        with self.db.connection() as conn:
            conn.execute("UPDATE extracted_poses SET is_sample_pose = 1 WHERE player_id = ?",
                        (player2_id,))

        player2_samples = self.db.get_sample_poses(player_name="Player 2")

        self.assertEqual(len(player2_samples), 1)

    def test_get_sample_poses_filter_by_stroke(self):
        """Test filtering sample poses by stroke type"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Add forehand sample pose
        pose_data = self._create_test_pose_data(video_id, player_id)
        pose_data.stroke_type = "forehand"
        self.db.add_pose(pose_data)
        with self.db.connection() as conn:
            conn.execute("UPDATE extracted_poses SET is_sample_pose = 1 WHERE stroke_type = 'forehand'")

        # Add serve sample pose
        pose_data = self._create_test_pose_data(video_id, player_id, frame_num=10)
        pose_data.stroke_type = "serve"
        self.db.add_pose(pose_data)
        with self.db.connection() as conn:
            conn.execute("UPDATE extracted_poses SET is_sample_pose = 1 WHERE stroke_type = 'serve'")

        forehand_samples = self.db.get_sample_poses(stroke_type="forehand")
        serve_samples = self.db.get_sample_poses(stroke_type="serve")

        self.assertEqual(len(forehand_samples), 1)
        self.assertEqual(len(serve_samples), 1)

    # =========================================================================
    # Similarity Search Tests
    # =========================================================================

    def test_find_similar_poses(self):
        """Test finding similar poses"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Add sample pose with known landmarks
        landmarks = np.linspace(0, 1, 66).reshape(33, 2).astype(np.float32)
        pose_data = self._create_test_pose_data(video_id, player_id)
        pose_data.landmarks = landmarks
        self.db.add_pose(pose_data)

        # Mark as sample pose
        with self.db.connection() as conn:
            conn.execute("UPDATE extracted_poses SET is_sample_pose = 1")

        # Search with similar landmarks
        query_landmarks = landmarks + np.random.normal(0, 0.05, (33, 2))
        similar = self.db.find_similar_poses(query_landmarks, top_k=5)

        self.assertEqual(len(similar), 1)
        self.assertGreater(similar[0][1], 0)  # Similarity > 0

    def test_find_similar_poses_exact_match(self):
        """Test finding exact match returns high similarity"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")

        # Create and store pose
        landmarks = np.random.rand(33, 2).astype(np.float32)
        pose_data = self._create_test_pose_data(video_id, player_id)
        pose_data.landmarks = landmarks
        self.db.add_pose(pose_data)

        # Mark as sample pose
        with self.db.connection() as conn:
            conn.execute("UPDATE extracted_poses SET is_sample_pose = 1")

        # Search with same landmarks
        similar = self.db.find_similar_poses(landmarks, top_k=1)

        # Should have very high similarity (>0.9 for exact match)
        self.assertGreater(similar[0][1], 0.9)

    def test_calculate_pose_similarity_identical(self):
        """Test pose similarity for identical landmarks"""
        landmarks = np.random.rand(33, 2)

        similarity = self.db.calculate_pose_similarity(landmarks, landmarks)

        self.assertEqual(similarity, 1.0)

    def test_calculate_pose_similarity_different(self):
        """Test pose similarity for different landmarks"""
        landmarks1 = np.zeros((33, 2))
        landmarks2 = np.ones((33, 2))

        similarity = self.db.calculate_pose_similarity(landmarks1, landmarks2)

        self.assertGreater(similarity, 0)  # Should still have some similarity
        self.assertLess(similarity, 1.0)  # Should not be perfect

    def test_calculate_angle_similarity(self):
        """Test angle similarity calculation"""
        angles1 = {'elbow': 90, 'shoulder': 45, 'knee': 120}
        angles2 = {'elbow': 92, 'shoulder': 47, 'knee': 118}

        similarity = self.db.calculate_angle_similarity(angles1, angles2)

        self.assertGreater(similarity, 0.9)  # Should be high similarity

    def test_calculate_angle_similarity_empty(self):
        """Test angle similarity with empty inputs"""
        similarity = self.db.calculate_angle_similarity({}, {})
        self.assertEqual(similarity, 0.0)

    # =========================================================================
    # Gesture Sequence Tests
    # =========================================================================

    def test_add_gesture_sequence(self):
        """Test adding a gesture sequence"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_id = self.db.add_pose(self._create_test_pose_data(video_id, player_id))

        sequence_id = self.db.add_gesture_sequence(
            pose_id=pose_id,
            video_id=video_id,
            player_id=player_id,
            sequence_type="forehand",
            start_frame=0,
            end_frame=50,
            key_frame=25
        )

        self.assertIsInstance(sequence_id, int)
        self.assertGreater(sequence_id, 0)

    def test_add_gesture_sequence_with_trajectory(self):
        """Test adding gesture sequence with trajectory data"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_id = self.db.add_pose(self._create_test_pose_data(video_id, player_id))

        trajectory = np.random.rand(50, 2)
        velocity = np.random.rand(50, 33, 2)
        acceleration = np.random.rand(50)

        sequence_id = self.db.add_gesture_sequence(
            pose_id=pose_id,
            video_id=video_id,
            player_id=player_id,
            sequence_type="serve",
            start_frame=0,
            end_frame=50,
            trajectory_data=trajectory,
            velocity_profile=velocity,
            acceleration_profile=acceleration
        )

        self.assertIsInstance(sequence_id, int)

    def test_get_sequences_by_player(self):
        """Test getting sequences for a player"""
        player1_id = self.db.add_player("Player 1")
        player2_id = self.db.add_player("Player 2")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_id = self.db.add_pose(self._create_test_pose_data(video_id, player1_id))

        # Add sequences for both players
        self.db.add_gesture_sequence(
            pose_id=pose_id, video_id=video_id, player_id=player1_id,
            sequence_type="forehand", start_frame=0, end_frame=30
        )
        self.db.add_gesture_sequence(
            pose_id=pose_id, video_id=video_id, player_id=player2_id,
            sequence_type="backhand", start_frame=50, end_frame=80
        )

        player1_seqs = self.db.get_sequences_by_player(player1_id)
        player2_seqs = self.db.get_sequences_by_player(player2_id)

        self.assertEqual(len(player1_seqs), 1)
        self.assertEqual(len(player2_seqs), 1)

    def test_get_sequences_by_player_with_type_filter(self):
        """Test filtering sequences by type"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_id = self.db.add_pose(self._create_test_pose_data(video_id, player_id))

        # Add different sequence types
        self.db.add_gesture_sequence(
            pose_id=pose_id, video_id=video_id, player_id=player_id,
            sequence_type="forehand", start_frame=0, end_frame=30
        )
        self.db.add_gesture_sequence(
            pose_id=pose_id, video_id=video_id, player_id=player_id,
            sequence_type="serve", start_frame=50, end_frame=80
        )

        forehand_seqs = self.db.get_sequences_by_player(player_id, sequence_type="forehand")
        serve_seqs = self.db.get_sequences_by_player(player_id, sequence_type="serve")

        self.assertEqual(len(forehand_seqs), 1)
        self.assertEqual(len(serve_seqs), 1)

    # =========================================================================
    # Comparison Results Tests
    # =========================================================================

    def test_save_comparison_result(self):
        """Test saving a comparison result"""
        player_id = self.db.add_player("Test Player")
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_id = self.db.add_pose(self._create_test_pose_data(video_id, player_id))

        result_id = self.db.save_comparison_result(
            input_video_path="/input/test.mp4",
            matched_player_id=player_id,
            similarity_score=0.85,
            input_pose_id=pose_id,
            pose_similarity=0.82,
            angle_similarity=0.88
        )

        self.assertIsInstance(result_id, int)

    def test_save_comparison_result_with_recommendations(self):
        """Test saving comparison result with recommendations"""
        player_id = self.db.add_player("Test Player")

        recommendations = [
            "Adjust your elbow angle",
            "Focus on follow-through"
        ]

        result_id = self.db.save_comparison_result(
            input_video_path="/input/test.mp4",
            matched_player_id=player_id,
            similarity_score=0.75,
            recommendations=recommendations
        )

        self.assertIsInstance(result_id, int)

    def test_get_comparison_history(self):
        """Test getting comparison history"""
        player_id = self.db.add_player("Test Player")

        # Save multiple results
        self.db.save_comparison_result(
            input_video_path="/input/test1.mp4",
            matched_player_id=player_id,
            similarity_score=0.80
        )
        self.db.save_comparison_result(
            input_video_path="/input/test2.mp4",
            matched_player_id=player_id,
            similarity_score=0.90
        )

        all_results = self.db.get_comparison_history()
        filtered_results = self.db.get_comparison_history("/input/test1.mp4")

        self.assertEqual(len(all_results), 2)
        self.assertEqual(len(filtered_results), 1)

    # =========================================================================
    # Statistics Tests
    # =========================================================================

    def test_get_statistics(self):
        """Test getting database statistics"""
        # Add some data
        player_id = self.db.add_player("Test Player", is_professional=True)
        video_id = self.db.add_video("test.mp4", "/path/to/test.mp4")
        pose_id = self.db.add_pose(self._create_test_pose_data(video_id, player_id))

        stats = self.db.get_statistics()

        self.assertEqual(stats['total_players'], 1)
        self.assertEqual(stats['professional_players'], 1)
        self.assertEqual(stats['total_videos'], 1)
        self.assertGreaterEqual(stats['total_landmarks'], 33)

    def test_get_statistics_empty_database(self):
        """Test statistics on empty database"""
        stats = self.db.get_statistics()

        self.assertEqual(stats['total_players'], 0)
        self.assertEqual(stats['total_videos'], 0)
        self.assertEqual(stats['total_poses'], 0)


class TestPoseData(unittest.TestCase):
    """Test cases for PoseData dataclass"""

    def test_pose_data_creation(self):
        """Test creating PoseData with minimal fields"""
        landmarks = np.random.rand(33, 2)

        pose = PoseData(
            id=None,
            video_id=1,
            frame_number=0,
            player_id=1,
            landmarks=landmarks
        )

        self.assertEqual(pose.video_id, 1)
        self.assertEqual(pose.confidence, 1.0)  # Default value
        self.assertIsNone(pose.stroke_type)

    def test_pose_data_with_all_fields(self):
        """Test creating PoseData with all fields"""
        landmarks = np.random.rand(33, 2)
        angles = {'elbow': 90, 'shoulder': 45}
        bbox = (0.1, 0.2, 0.3, 0.4)

        pose = PoseData(
            id=1,
            video_id=2,
            frame_number=10,
            player_id=3,
            landmarks=landmarks,
            joint_angles=angles,
            bbox=bbox,
            stroke_type="forehand",
            confidence=0.95,
            timestamp_ms=333.33
        )

        self.assertEqual(pose.id, 1)
        self.assertEqual(pose.joint_angles['elbow'], 90)
        self.assertEqual(pose.bbox, bbox)
        self.assertEqual(pose.stroke_type, "forehand")


if __name__ == '__main__':
    unittest.main()
