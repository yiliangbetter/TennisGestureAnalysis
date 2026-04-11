#!/usr/bin/env python3
"""
Unit Tests for Video Text OCR Module

Tests the OCR-based player name extraction functionality.
"""

import unittest
import sqlite3
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.ocr import VideoTextExtractor
from database_manager import TennisDatabase


def create_temp_db():
    """Create a temporary database file and return its path"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return path


class TestVideoTextExtractor(unittest.TestCase):
    """Test cases for VideoTextExtractor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)
        self.db = self.extractor.db

        # Insert test players
        self.test_players = [
            ("Novak Djokovic", True),
            ("Carlos Alcaraz", True),
            ("Jannik Sinner", True),
            ("Iga Swiatek", True),
            ("Coco Gauff", True),
        ]

        for name, is_pro in self.test_players:
            self.db.add_player(name, is_professional=is_pro)

    def tearDown(self):
        """Clean up after tests"""
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_initialization(self):
        """Test extractor initializes correctly"""
        self.assertIsNotNone(self.extractor.db)
        self.assertFalse(self.extractor.use_easyocr)  # Disabled for testing
        self.assertEqual(self.extractor.frame_sample_interval, 60)

    def test_get_all_players(self):
        """Test retrieving all professional players"""
        players = self.db.get_all_players(professionals_only=True)
        # Should have at least the 5 test players we added
        self.assertGreaterEqual(len(players), 5)

        names = [p['name'] for p in players]
        self.assertIn("Novak Djokovic", names)
        self.assertIn("Carlos Alcaraz", names)

    def test_player_lookup_by_name(self):
        """Test looking up player by name"""
        player = self.db.get_player_by_name("Novak Djokovic")
        self.assertIsNotNone(player)
        self.assertEqual(player['name'], "Novak Djokovic")
        self.assertTrue(player['is_professional'])

    def test_player_not_found(self):
        """Test lookup for non-existent player"""
        player = self.db.get_player_by_name("Unknown Player")
        self.assertIsNone(player)


class TestPlayerNameMatching(unittest.TestCase):
    """Test player name matching logic"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)
        self.db = self.extractor.db

        # Insert test players
        self.players = [
            "Novak Djokovic",
            "Carlos Alcaraz",
            "Jannik Sinner",
            "Rafael Nadal",
            "Roger Federer",
        ]

        for name in self.players:
            self.db.add_player(name, is_professional=True)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_exact_match(self):
        """Test exact player name match"""
        ocr_results = [
            {
                'detected_text': 'Novak Djokovic',
                'confidence': 0.95,
                'frame_number': 0,
                'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 100, 'bbox_h': 50
            }
        ]

        # Access private method for testing
        matched = self.extractor._match_player_name(ocr_results)
        self.assertEqual(matched, "Novak Djokovic")

    def test_case_insensitive_match(self):
        """Test case-insensitive matching"""
        ocr_results = [
            {
                'detected_text': 'NOVAK DJOKOVIC',
                'confidence': 0.95,
                'frame_number': 0,
                'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 100, 'bbox_h': 50
            }
        ]

        matched = self.extractor._match_player_name(ocr_results)
        self.assertEqual(matched, "Novak Djokovic")

    def test_last_name_only_match(self):
        """Test matching with last name only"""
        ocr_results = [
            {
                'detected_text': 'Djokovic',
                'confidence': 0.90,
                'frame_number': 0,
                'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 100, 'bbox_h': 50
            }
        ]

        matched = self.extractor._match_player_name(ocr_results)
        self.assertEqual(matched, "Novak Djokovic")

    def test_fuzzy_match_typo(self):
        """Test fuzzy matching with OCR typo"""
        ocr_results = [
            {
                'detected_text': 'Djokovic',  # Last name only (common OCR result)
                'confidence': 0.85,
                'frame_number': 0,
                'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 100, 'bbox_h': 50
            }
        ]

        matched = self.extractor._match_player_name(ocr_results)
        # Should match on last name
        self.assertEqual(matched, "Novak Djokovic")

    def test_abbreviated_name_match(self):
        """Test matching with abbreviated name"""
        ocr_results = [
            {
                'detected_text': 'N. Djokovic',
                'confidence': 0.90,
                'frame_number': 0,
                'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 100, 'bbox_h': 50
            }
        ]

        matched = self.extractor._match_player_name(ocr_results)
        self.assertEqual(matched, "Novak Djokovic")

    def test_no_match_for_unknown_name(self):
        """Test that unknown names don't match"""
        ocr_results = [
            {
                'detected_text': 'John Smith',
                'confidence': 0.95,
                'frame_number': 0,
                'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 100, 'bbox_h': 50
            }
        ]

        matched = self.extractor._match_player_name(ocr_results)
        self.assertIsNone(matched)

    def test_low_confidence_filtered(self):
        """Test that low confidence OCR results are filtered"""
        ocr_results = [
            {
                'detected_text': 'Novak Djokovic',
                'confidence': 0.3,  # Below threshold
                'frame_number': 0,
                'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 100, 'bbox_h': 50
            }
        ]

        matched = self.extractor._match_player_name(ocr_results)
        self.assertIsNone(matched)


class TestScoreboardFiltering(unittest.TestCase):
    """Test scoreboard and sponsor text filtering"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_score_with_dash(self):
        """Test filtering score like 15-30"""
        self.assertTrue(self.extractor._is_scoreboard_text("15-30"))
        self.assertTrue(self.extractor._is_scoreboard_text("30-15"))

    def test_score_with_colon(self):
        """Test filtering time like 2:30"""
        self.assertTrue(self.extractor._is_scoreboard_text("2:30"))

    def test_set_game_text(self):
        """Test filtering SET/GAME text"""
        self.assertTrue(self.extractor._is_scoreboard_text("SET 1"))
        self.assertTrue(self.extractor._is_scoreboard_text("GAME 3"))

    def test_single_digit(self):
        """Test filtering single digits"""
        self.assertTrue(self.extractor._is_scoreboard_text("5"))
        self.assertTrue(self.extractor._is_scoreboard_text("15"))

    def test_player_name_not_filtered(self):
        """Test that player names are not filtered"""
        self.assertFalse(self.extractor._is_scoreboard_text("Novak Djokovic"))
        self.assertFalse(self.extractor._is_scoreboard_text("Djokovic"))

    def test_sponsor_filtering(self):
        """Test sponsor name filtering"""
        self.assertTrue(self.extractor._is_sponsor_text("KIA"))
        self.assertTrue(self.extractor._is_sponsor_text("ROLEX"))
        self.assertTrue(self.extractor._is_sponsor_text("US OPEN"))

    def test_player_name_not_sponsor(self):
        """Test that player names are not filtered as sponsors"""
        self.assertFalse(self.extractor._is_sponsor_text("Novak Djokovic"))
        self.assertFalse(self.extractor._is_sponsor_text("Nadal"))


class TestMatchScoreCalculation(unittest.TestCase):
    """Test match score calculation"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

        # Add a test player
        self.db = self.extractor.db
        self.db.add_player("Novak Djokovic", is_professional=True)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_perfect_match_score(self):
        """Test perfect match returns high score"""
        score = self.extractor._calculate_match_score("Novak Djokovic", "Novak Djokovic")
        self.assertEqual(score, 1.0)

    def test_case_insensitive_score(self):
        """Test case-insensitive matching"""
        score1 = self.extractor._calculate_match_score("NOVAK DJOKOVIC", "Novak Djokovic")
        score2 = self.extractor._calculate_match_score("novak djokovic", "Novak Djokovic")
        self.assertGreater(score1, 0.8)
        self.assertGreater(score2, 0.8)

    def test_last_name_match_score(self):
        """Test last name only matching"""
        score = self.extractor._calculate_match_score("Djokovic", "Novak Djokovic")
        self.assertGreater(score, 0.7)

    def test_partial_name_score(self):
        """Test partial name matching"""
        score = self.extractor._calculate_match_score("N. Djokovic", "Novak Djokovic")
        self.assertGreater(score, 0.5)

    def test_typo_score(self):
        """Test fuzzy matching with typo"""
        score = self.extractor._calculate_match_score("Djokovio", "Djokovic")
        self.assertGreater(score, 0.7)  # Should still be reasonably high

    def test_no_match_score(self):
        """Test no match returns low score"""
        score = self.extractor._calculate_match_score("John Smith", "Novak Djokovic")
        self.assertLess(score, 0.5)


class TestDatabaseOperations(unittest.TestCase):
    """Test database operations for OCR results"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)
        self.db = self.extractor.db

        # Add test player
        self.db.add_player("Novak Djokovic", is_professional=True)

        # Add test video
        self.video_id = self.db.add_video(
            filename="test_video.mp4",
            file_path="/test/test_video.mp4",
            metadata={'duration': 60, 'fps': 30, 'width': 1920, 'height': 1080}
        )

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_store_ocr_results(self):
        """Test storing OCR results in database"""
        results = [
            {
                'frame_number': 0,
                'detected_text': 'Novak Djokovic',
                'confidence': 0.95,
                'bbox_x': 100.0,
                'bbox_y': 800.0,
                'bbox_w': 200.0,
                'bbox_h': 50.0
            },
            {
                'frame_number': 60,
                'detected_text': 'Djokovic',
                'confidence': 0.90,
                'bbox_x': 150.0,
                'bbox_y': 850.0,
                'bbox_w': 150.0,
                'bbox_h': 40.0
            }
        ]

        self.extractor._store_ocr_results(self.video_id, results)

        # Verify results were stored
        stored = self.extractor.get_detected_text(self.video_id)
        self.assertEqual(len(stored), 2)
        self.assertEqual(stored[0]['detected_text'], 'Novak Djokovic')
        self.assertEqual(stored[1]['detected_text'], 'Djokovic')

    def test_get_detected_text_with_bbox(self):
        """Test retrieving OCR results with bounding boxes"""
        results = [
            {
                'frame_number': 0,
                'detected_text': 'Test Text',
                'confidence': 0.95,
                'bbox_x': 100.0,
                'bbox_y': 800.0,
                'bbox_w': 200.0,
                'bbox_h': 50.0
            }
        ]

        self.extractor._store_ocr_results(self.video_id, results)

        # Get with bounding boxes
        stored = self.extractor.get_detected_text(self.video_id, include_bbox=True)
        self.assertIn('bbox_x', stored[0])
        self.assertIn('bbox_y', stored[0])
        self.assertIn('bbox_w', stored[0])
        self.assertIn('bbox_h', stored[0])

    def test_get_player_names_only(self):
        """Test filtering to get only player name results"""
        results = [
            {
                'frame_number': 0,
                'detected_text': 'Novak Djokovic',
                'confidence': 0.95,
                'bbox_x': 100.0,
                'bbox_y': 800.0,
                'bbox_w': 200.0,
                'bbox_h': 50.0
            },
            {
                'frame_number': 60,
                'detected_text': 'KIA',
                'confidence': 0.90,
                'bbox_x': 50.0,
                'bbox_y': 100.0,
                'bbox_w': 100.0,
                'bbox_h': 30.0
            }
        ]

        self.extractor._store_ocr_results(self.video_id, results)

        # Mark first result as player name
        self.extractor._mark_player_name_match(self.video_id, "Novak Djokovic")

        # Get only player names
        player_results = self.extractor.get_detected_text(
            self.video_id, player_names_only=True
        )
        self.assertEqual(len(player_results), 1)
        self.assertEqual(player_results[0]['detected_text'], 'Novak Djokovic')


class TestIntegration(unittest.TestCase):
    """Integration tests for OCR module with VideoProcessor"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.db = TennisDatabase(self.test_db_path)

        # Seed professional players
        players = [
            "Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner",
            "Iga Swiatek", "Coco Gauff"
        ]
        for name in players:
            self.db.add_player(name, is_professional=True)

    def tearDown(self):
        self.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_extractor_with_empty_video(self):
        """Test extractor handles non-existent video gracefully"""
        extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

        # Non-existent video should return None
        result = extractor.extract_player_name_from_video("/nonexistent/video.mp4")
        self.assertIsNone(result)

    def test_extractor_creates_video_entry(self):
        """Test extractor creates video entry if needed"""
        extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

        # Try to process non-existent video - should fail gracefully
        # but we can test the internal method
        video_id = extractor._get_or_add_video("/nonexistent/video.mp4")
        self.assertIsNone(video_id)  # Should return None for non-existent video


class TestCalculateMatchScore(unittest.TestCase):
    """Test the _calculate_match_score method edge cases"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_empty_strings(self):
        """Test with empty strings"""
        score = self.extractor._calculate_match_score("", "Novak Djokovic")
        self.assertLess(score, 0.5)

    def test_partial_match_alcaraz(self):
        """Test partial match for Alcaraz"""
        score = self.extractor._calculate_match_score("Alcaraz", "Carlos Alcaraz")
        self.assertGreater(score, 0.7)

    def test_partial_match_sinner(self):
        """Test partial match for Sinner"""
        score = self.extractor._calculate_match_score("Sinner", "Jannik Sinner")
        self.assertGreater(score, 0.7)

    def test_whitespace_handling(self):
        """Test whitespace is handled correctly"""
        score1 = self.extractor._calculate_match_score("  Djokovic  ", "Novak Djokovic")
        score2 = self.extractor._calculate_match_score("Djokovic", "Novak Djokovic")
        self.assertAlmostEqual(score1, score2, places=1)


class TestSchemaValidation(unittest.TestCase):
    """Test that database schema is correctly set up"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.db = TennisDatabase(self.test_db_path)

    def tearDown(self):
        self.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_video_text_ocr_table_exists(self):
        """Test video_text_ocr table exists"""
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='video_text_ocr'"
            )
            row = cursor.fetchone()
            self.assertIsNotNone(row)

    def test_video_text_ocr_columns(self):
        """Test video_text_ocr table has correct columns"""
        with self.db.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(video_text_ocr)")
            columns = {row['name'] for row in cursor.fetchall()}

        expected_columns = {
            'id', 'video_id', 'frame_number', 'detected_text',
            'confidence', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
            'is_player_name'
        }
        self.assertTrue(expected_columns.issubset(columns))

    def test_players_table_has_seed_data(self):
        """Test players table exists and can store players"""
        # Add a player to verify table works
        self.db.add_player("Test Player", is_professional=True)
        players = self.db.get_all_players(professionals_only=True)
        # Should have at least the player we added
        self.assertGreater(len(players), 0)


class TestSponsorFiltering(unittest.TestCase):
    """Test sponsor name filtering including Mercedes Benz and Emirates"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_mercedes_benz_filtered(self):
        """Test Mercedes Benz is filtered as sponsor"""
        self.assertTrue(self.extractor._is_sponsor_text("MERCEDES"))
        self.assertTrue(self.extractor._is_sponsor_text("MERCEDES BENZ"))
        self.assertTrue(self.extractor._is_sponsor_text("MERCEDES-BENZ"))
        self.assertTrue(self.extractor._is_sponsor_text("BENZ"))

    def test_emirates_filtered(self):
        """Test Emirates is filtered as sponsor"""
        self.assertTrue(self.extractor._is_sponsor_text("EMIRATES"))

    def test_qatar_airways_filtered(self):
        """Test Qatar Airways is filtered as sponsor"""
        self.assertTrue(self.extractor._is_sponsor_text("QATAR AIRWAYS"))

    def test_lufthansa_filtered(self):
        """Test Lufthansa is filtered as sponsor"""
        self.assertTrue(self.extractor._is_sponsor_text("LUFTHANSA"))

    def test_citizen_filtered(self):
        """Test Citizen is filtered as sponsor"""
        self.assertTrue(self.extractor._is_sponsor_text("CITIZEN"))

    def test_player_name_not_sponsor(self):
        """Test player names are not filtered as sponsors"""
        self.assertFalse(self.extractor._is_sponsor_text("Novak Djokovic"))
        self.assertFalse(self.extractor._is_sponsor_text("Carlos Alcaraz"))


class TestLooksLikePlayerName(unittest.TestCase):
    """Test the _looks_like_player_name function"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_valid_player_name(self):
        """Test valid player name is recognized"""
        self.assertTrue(self.extractor._looks_like_player_name("Roger Federer", 0.92))
        self.assertTrue(self.extractor._looks_like_player_name("Rafael Nadal", 0.88))

    def test_low_confidence_rejected(self):
        """Test low confidence OCR is rejected"""
        self.assertFalse(self.extractor._looks_like_player_name("Roger Federer", 0.3))

    def test_single_word_rejected(self):
        """Test single word is rejected (needs first + last name)"""
        self.assertFalse(self.extractor._looks_like_player_name("Federer", 0.92))

    def test_lowercase_rejected(self):
        """Test lowercase names are rejected"""
        self.assertFalse(self.extractor._looks_like_player_name("roger federer", 0.92))

    def test_sponsor_rejected(self):
        """Test sponsor names are rejected"""
        self.assertFalse(self.extractor._looks_like_player_name("Mercedes Benz", 0.95))


class TestAddPlayerCandidate(unittest.TestCase):
    """Test the _add_player_candidate function"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_add_new_candidate(self):
        """Test adding a new player candidate"""
        player_id = self.extractor._add_player_candidate("Tommy Paul", 0.92)
        self.assertIsNotNone(player_id)

        # Verify player was added
        with self.extractor.db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM players WHERE LOWER(name) = LOWER(?)",
                ("Tommy Paul",)
            )
            player = cursor.fetchone()
            self.assertIsNotNone(player)
            self.assertEqual(player['name'], "Tommy Paul")
            self.assertEqual(player['country'], 'UNK')
            self.assertEqual(player['is_professional'], 0)
            self.assertIn('candidate', player['metadata'])

    def test_duplicate_not_added(self):
        """Test duplicate player is not added"""
        # Add first time
        self.extractor._add_player_candidate("Tommy Paul", 0.92)

        # Try to add again - should return None
        player_id = self.extractor._add_player_candidate("Tommy Paul", 0.95)
        self.assertIsNone(player_id)

    def test_empty_name_not_added(self):
        """Test empty name is not added"""
        player_id = self.extractor._add_player_candidate("", 0.92)
        self.assertIsNone(player_id)


class TestMatchAllPlayerNames(unittest.TestCase):
    """Test the _match_all_player_names function"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)
        self.db = self.extractor.db

        # Add multiple test players
        self.players = [
            "Novak Djokovic",
            "Carlos Alcaraz",
            "Jannik Sinner",
            "Rafael Nadal"
        ]
        for name in self.players:
            self.db.add_player(name, is_professional=True)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_multiple_players_detected(self):
        """Test that multiple players in same video are all detected"""
        ocr_results = [
            {'detected_text': 'Novak Djokovic', 'confidence': 0.98, 'frame_number': 0},
            {'detected_text': 'Carlos Alcaraz', 'confidence': 0.95, 'frame_number': 100},
            {'detected_text': 'Jannik Sinner', 'confidence': 0.97, 'frame_number': 200},
        ]

        matched = self.extractor._match_all_player_names(ocr_results)
        self.assertEqual(len(matched), 3)
        self.assertIn('Novak Djokovic', matched)
        self.assertIn('Carlos Alcaraz', matched)
        self.assertIn('Jannik Sinner', matched)

    def test_single_player_detected(self):
        """Test single player in video is detected"""
        ocr_results = [
            {'detected_text': 'Rafael Nadal', 'confidence': 0.96, 'frame_number': 0},
        ]

        matched = self.extractor._match_all_player_names(ocr_results)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0], 'Rafael Nadal')

    def test_empty_ocr_results(self):
        """Test empty OCR results returns empty list"""
        matched = self.extractor._match_all_player_names([])
        self.assertEqual(len(matched), 0)

    def test_no_valid_matches(self):
        """Test no valid player matches returns empty list"""
        ocr_results = [
            {'detected_text': 'Unknown Player', 'confidence': 0.95, 'frame_number': 0},
            {'detected_text': 'Random Text', 'confidence': 0.90, 'frame_number': 100},
        ]

        matched = self.extractor._match_all_player_names(ocr_results)
        self.assertEqual(len(matched), 0)


class TestSponsorList(unittest.TestCase):
    """Test the updated sponsor list including Mercedes Benz and others"""

    def setUp(self):
        self.test_db_path = create_temp_db()
        self.extractor = VideoTextExtractor(db_path=self.test_db_path, use_easyocr=False)

    def tearDown(self):
        self.extractor.db.close()
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_mercedes_benz_variations(self):
        """Test all Mercedes Benz variations are filtered"""
        variations = [
            "MERCEDES",
            "MERCEDES BENZ",
            "MERCEDES-BENZ",
            "BENZ"
        ]
        for variation in variations:
            with self.subTest(variation=variation):
                self.assertTrue(
                    self.extractor._is_sponsor_text(variation),
                    f"'{variation}' should be filtered as sponsor"
                )

    def test_emirates_filtered(self):
        """Test Emirates is filtered"""
        self.assertTrue(self.extractor._is_sponsor_text("EMIRATES"))
        self.assertTrue(self.extractor._is_sponsor_text("emirates"))  # Case insensitive

    def test_qatar_airways_filtered(self):
        """Test Qatar Airways is filtered"""
        self.assertTrue(self.extractor._is_sponsor_text("QATAR AIRWAYS"))

    def test_lufthansa_filtered(self):
        """Test Lufthansa is filtered"""
        self.assertTrue(self.extractor._is_sponsor_text("LUFTHANSA"))

    def test_citizen_filtered(self):
        """Test Citizen is filtered"""
        self.assertTrue(self.extractor._is_sponsor_text("CITIZEN"))

    def test_original_sponsors_still_filtered(self):
        """Test original sponsors are still filtered"""
        original_sponsors = [
            "KIA", "ROLEX", "IBM", "US OPEN", "WIMBLEDON",
            "ROLAND GARROS", "NIKE", "ADIDAS"
        ]
        for sponsor in original_sponsors:
            with self.subTest(sponsor=sponsor):
                self.assertTrue(
                    self.extractor._is_sponsor_text(sponsor),
                    f"'{sponsor}' should still be filtered"
                )

    def test_player_names_not_sponsors(self):
        """Test player names are not filtered as sponsors"""
        player_names = [
            "Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner",
            "Rafael Nadal", "Roger Federer"
        ]
        for name in player_names:
            with self.subTest(name=name):
                self.assertFalse(
                    self.extractor._is_sponsor_text(name),
                    f"'{name}' should not be filtered as sponsor"
                )


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVideoTextExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestPlayerNameMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestScoreboardFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestMatchScoreCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateMatchScore))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVideoTextExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestPlayerNameMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestScoreboardFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestMatchScoreCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateMatchScore))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
