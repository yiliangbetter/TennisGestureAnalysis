#!/usr/bin/env python3
"""
Unit tests for Video Processor module.

Tests cover:
- Player name extraction from filenames
- Stroke type detection from filenames
- Video processor initialization
- Comparison result formatting
"""

import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from video_processor import VideoProcessor
from database_manager import TennisDatabase, PoseData


class TestPlayerNameExtraction(unittest.TestCase):
    """Test player name extraction from filenames"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.processor = VideoProcessor(self.db_path)

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_extract_djokovic_name(self):
        """Test extracting Djokovic name"""
        filename = "tennis-djokovic-forehand-Novak Djokovic Slow Motion.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Novak Djokovic")

    def test_extract_alcaraz_name(self):
        """Test extracting Alcaraz name"""
        filename = "tennis-pro-forehead-Alcaraz Compilation.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Carlos Alcaraz")

    def test_extract_sinner_name(self):
        """Test extracting Sinner name"""
        filename = "sinner-serve-technique.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Jannik Sinner")

    def test_extract_zverev_name(self):
        """Test extracting Zverev name"""
        filename = "zverev-backhand.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Alexander Zverev")

    def test_extract_rublev_name(self):
        """Test extracting Rublev name"""
        filename = "rublev-forehand.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Andrey Rublev")

    def test_extract_shelton_name(self):
        """Test extracting Shelton name"""
        filename = "shelton-serve.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Ben Shelton")

    def test_extract_nadal_name(self):
        """Test extracting Nadal name"""
        filename = "nadal-forehand-topspin.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Rafael Nadal")

    def test_extract_federer_name(self):
        """Test extracting Federer name"""
        filename = "federer-slice-backhand.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Roger Federer")

    def test_extract_serena_name(self):
        """Test extracting Serena name"""
        filename = "serena-williams-serve.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Serena Williams")

    def test_extract_swiatek_name(self):
        """Test extracting Swiatek name"""
        filename = "swiatek-forehand.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Iga Swiatek")

    def test_extract_gauff_name(self):
        """Test extracting Gauff name"""
        filename = "gauff-backhand.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "Coco Gauff")

    def test_fallback_to_filename_stem(self):
        """Test fallback when no known player found"""
        filename = "unknown-player-tennis.mp4"
        name = self.processor._extract_player_name(filename)
        # Should return title case of filename
        self.assertIn("Unknown", name) or self.assertIn("Player", name) or self.assertIn("Tennis", name)

    def test_capitalized_name_extraction(self):
        """Test extracting capitalized names from filename"""
        # Use a filename without known player keys to test regex extraction
        filename = "tennis-match-John Smith vs Jane Doe.mp4"
        name = self.processor._extract_player_name(filename)
        self.assertEqual(name, "John Smith")


class TestStrokeTypeExtraction(unittest.TestCase):
    """Test stroke type extraction from filenames"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.processor = VideoProcessor(self.db_path)

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_extract_forehand(self):
        """Test extracting forehand stroke type"""
        filename = "djokovic-forehand-slow-mo.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "forehand")

    def test_extract_backhand(self):
        """Test extracting backhand stroke type"""
        filename = "federer-backhand-slice.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "backhand")

    def test_extract_serve(self):
        """Test extracting serve stroke type"""
        filename = "isner-serve-technique.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "serve")

    def test_extract_volley(self):
        """Test extracting volley stroke type"""
        filename = "mcenroe-volley-net.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "volley")

    def test_extract_fh_abbreviation(self):
        """Test extracting forehand from fh abbreviation"""
        filename = "player-fh-swing.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "forehand")

    def test_extract_bh_abbreviation(self):
        """Test extracting backhand from bh abbreviation"""
        filename = "player-bh-return.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "backhand")

    def test_chinese_forehand(self):
        """Test extracting forehand from Chinese characters"""
        filename = "tennis-正手-technique.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "forehand")

    def test_chinese_backhand(self):
        """Test extracting backhand from Chinese characters"""
        filename = "tennis-反手-slice.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "backhand")

    def test_chinese_serve(self):
        """Test extracting serve from Chinese characters"""
        filename = "tennis-发球-power.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "serve")

    def test_default_to_forehand(self):
        """Test defaulting to forehand when no stroke detected"""
        filename = "tennis-generic-swing.mp4"
        stroke = self.processor._extract_stroke_type(filename)
        self.assertEqual(stroke, "forehand")


class TestAngleMapping(unittest.TestCase):
    """Test joint angle mapping"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.processor = VideoProcessor(self.db_path)

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_map_all_angles(self):
        """Test mapping all 12 angle types"""
        angles_list = [90 + i for i in range(12)]

        angle_dict = self.processor._map_angles(angles_list)

        expected_keys = [
            'right_elbow_flexion', 'right_shoulder_abduction',
            'left_elbow_flexion', 'left_shoulder_abduction',
            'right_knee_flexion', 'right_hip_angle',
            'left_knee_flexion', 'left_hip_angle',
            'torso_rotation_right', 'torso_rotation_left',
            'body_lean_right', 'body_lean_left',
        ]

        self.assertEqual(len(angle_dict), 12)
        for key in expected_keys:
            self.assertIn(key, angle_dict)

    def test_map_partial_angles(self):
        """Test mapping when fewer than 12 angles provided"""
        angles_list = [90, 45, 60]  # Only 3 angles

        angle_dict = self.processor._map_angles(angles_list)

        self.assertEqual(angle_dict['right_elbow_flexion'], 90)
        self.assertEqual(angle_dict['right_shoulder_abduction'], 45)
        self.assertEqual(angle_dict['left_elbow_flexion'], 60)

    def test_map_empty_angles(self):
        """Test mapping empty angle list"""
        angles_list = []

        angle_dict = self.processor._map_angles(angles_list)

        # Should return 0.0 for all angles
        self.assertEqual(angle_dict['right_elbow_flexion'], 0.0)


class TestRecommendationGeneration(unittest.TestCase):
    """Test recommendation generation"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.processor = VideoProcessor(self.db_path)

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_high_similarity_recommendation(self):
        """Test recommendations for high similarity scores"""
        player_scores = {1: [0.92, 0.88, 0.95]}
        all_similarities = [0.92, 0.88, 0.95]

        recs = self.processor._generate_recommendations(player_scores, all_similarities)

        self.assertTrue(len(recs) > 0)
        self.assertTrue(any("Excellent" in rec for rec in recs))

    def test_good_similarity_recommendation(self):
        """Test recommendations for good similarity scores"""
        player_scores = {1: [0.72, 0.68, 0.75]}
        all_similarities = [0.72, 0.68, 0.75]

        recs = self.processor._generate_recommendations(player_scores, all_similarities)

        self.assertTrue(any("Good form" in rec for rec in recs))

    def test_medium_similarity_recommendation(self):
        """Test recommendations for medium similarity scores"""
        player_scores = {1: [0.52, 0.48, 0.55]}
        all_similarities = [0.52, 0.48, 0.55]

        recs = self.processor._generate_recommendations(player_scores, all_similarities)

        self.assertTrue(any("basic stance" in rec for rec in recs))

    def test_low_similarity_recommendation(self):
        """Test recommendations for low similarity scores"""
        player_scores = {1: [0.25, 0.30, 0.28]}
        all_similarities = [0.25, 0.30, 0.28]

        recs = self.processor._generate_recommendations(player_scores, all_similarities)

        self.assertTrue(any("fundamental technique" in rec for rec in recs))

    def test_low_frame_consistency_warning(self):
        """Test warning for low-scoring frames"""
        player_scores = {1: [0.4, 0.3]}
        all_similarities = [0.3, 0.4, 0.2, 0.3, 0.2]  # 60% below 0.5

        recs = self.processor._generate_recommendations(player_scores, all_similarities)

        self.assertTrue(any("30%" in rec or "consistency" in rec for rec in recs))

    def test_empty_similarities(self):
        """Test handling empty similarities"""
        player_scores = {}
        all_similarities = []

        recs = self.processor._generate_recommendations(player_scores, all_similarities)

        self.assertIsInstance(recs, list)


class TestVideoProcessorIntegration(unittest.TestCase):
    """Integration tests for VideoProcessor"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_processor_initialization(self):
        """Test processor initializes correctly"""
        processor = VideoProcessor(self.db_path)

        self.assertIsInstance(processor.db, TennisDatabase)
        self.assertIsNotNone(processor.analyzer)

    def test_database_created_on_init(self):
        """Test database file is created on initialization"""
        processor = VideoProcessor(self.db_path)

        self.assertTrue(os.path.exists(self.db_path))

    def test_processor_has_player_mapping(self):
        """Test processor has player name mappings"""
        processor = VideoProcessor(self.db_path)

        self.assertIn("djokovic", processor.PLAYER_NAME_MAPPING)
        self.assertIn("nadal", processor.PLAYER_NAME_MAPPING)

    def test_processor_has_stroke_keywords(self):
        """Test processor has stroke type keywords"""
        processor = VideoProcessor(self.db_path)

        self.assertIn("forehand", processor.STROKE_KEYWORDS)
        self.assertIn("backhand", processor.STROKE_KEYWORDS)
        self.assertIn("serve", processor.STROKE_KEYWORDS)


class TestPoseDataSerialization(unittest.TestCase):
    """Test PoseData serialization and deserialization"""

    def test_pose_data_numpy_landmarks(self):
        """Test PoseData handles numpy arrays correctly"""
        landmarks = np.random.rand(33, 2).astype(np.float32)

        pose = PoseData(
            id=1,
            video_id=2,
            frame_number=10,
            player_id=3,
            landmarks=landmarks
        )

        self.assertIsInstance(pose.landmarks, np.ndarray)
        self.assertEqual(pose.landmarks.shape, (33, 2))

    def test_pose_data_optional_fields(self):
        """Test PoseData optional fields default correctly"""
        landmarks = np.random.rand(33, 2)

        pose = PoseData(
            id=None,
            video_id=1,
            frame_number=0,
            player_id=1,
            landmarks=landmarks
        )

        self.assertIsNone(pose.id)
        self.assertEqual(pose.confidence, 1.0)
        self.assertIsNone(pose.stroke_type)
        self.assertIsNone(pose.bbox)
        self.assertEqual(pose.joint_angles, {})


if __name__ == '__main__':
    unittest.main()
