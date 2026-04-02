#!/usr/bin/env python3
"""
Additional Unit Tests for Video Text OCR Module

Tests for newly added functions:
- Sponsor list including Mercedes Benz, Emirates, etc.
- _looks_like_player_name()
- _add_player_candidate()
- _match_all_player_names()
"""

import unittest
import sqlite3
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.ocr import VideoTextExtractor
from database_manager import TennisDatabase


def create_temp_db():
    """Create a temporary database file and return its path"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return path


class TestSponsorList(unittest.TestCase):
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
        """Test Emirates is filtered as sponsor"""
        self.assertTrue(self.extractor._is_sponsor_text("EMIRATES"))
        self.assertTrue(self.extractor._is_sponsor_text("emirates"))  # Case insensitive

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

    def test_lowercase_accepted(self):
        """Test lowercase names are accepted (case insensitive)"""
        # The function is case-insensitive, so lowercase is accepted
        self.assertTrue(self.extractor._looks_like_player_name("roger federer", 0.92))

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
        # Use a unique name that definitely doesn't exist
        unique_name = "Test Player XYZ123"
        player_id = self.extractor._add_player_candidate(unique_name, 0.92)
        self.assertIsNotNone(player_id)

        # Verify player was added
        with self.extractor.db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM players WHERE LOWER(name) = LOWER(?)",
                (unique_name,)
            )
            player = cursor.fetchone()
            self.assertIsNotNone(player)
            self.assertEqual(player['name'], unique_name)
            self.assertEqual(player['country'], 'UNK')
            self.assertEqual(player['is_professional'], 0)
            self.assertIn('candidate', player['metadata'])

    def test_duplicate_not_added(self):
        """Test duplicate player is not added"""
        # Use a unique name
        unique_name = "Duplicate Test Player ABC"

        # Add first time
        first_id = self.extractor._add_player_candidate(unique_name, 0.92)
        self.assertIsNotNone(first_id)

        # Try to add again - should return None
        second_id = self.extractor._add_player_candidate(unique_name, 0.95)
        self.assertIsNone(second_id)

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


if __name__ == "__main__":
    # Run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSponsorList))
    suite.addTests(loader.loadTestsFromTestCase(TestLooksLikePlayerName))
    suite.addTests(loader.loadTestsFromTestCase(TestAddPlayerCandidate))
    suite.addTests(loader.loadTestsFromTestCase(TestMatchAllPlayerNames))

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

    sys.exit(0 if result.wasSuccessful() else 1)
