#!/usr/bin/env python3
"""
Video Text Extractor for Tennis Gesture Analysis

Extracts text from tennis video frames using OCR and identifies
player names by matching against a database of known players.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from difflib import SequenceMatcher

from database_manager import TennisDatabase


class VideoTextExtractor:
    """
    Extracts text from tennis videos and identifies player names.

    Workflow:
    1. Sample frames from video (every 60 frames ~ 1 per second at 60fps)
    2. Run OCR on sampled frames (full frame, no cropping assumptions)
    3. Store detected text with bounding boxes in database
    4. Match detected text against known player names
    5. Return identified player name
    """

    # Text patterns to filter out (not player names)
    SCOREBOARD_PATTERNS = [
        r'^\d+-\d+$',           # Score like "15-30"
        r'^\d+:\d+$',           # Time like "2:30"
        r'^SET\s*\d+$',         # "SET 1"
        r'^GAME\s*\d+$',        # "GAME 3"
        r'^\d{1,2}$',           # Single/double digit
    ]

    # Sponsor/brand names to filter out
    SPONSOR_NAMES = [
        'KIA', 'ROLEX', 'IBM', 'ANZ', 'US OPEN', 'WIMBLEDON',
        'ROLAND GARROS', 'FRENCH OPEN', 'AUSTRALIAN OPEN',
        'ATP', 'WTA', 'LACOSTE', 'NIKE', 'ADIDAS', 'Wilson',
        'HEAD', 'Babolat', 'YONEX', 'MIZUNO', 'PEPSI', 'HEINEKEN'
    ]

    # Minimum text length to consider as player name
    MIN_NAME_LENGTH = 3

    # Minimum confidence for OCR results
    MIN_OCR_CONFIDENCE = 0.5

    # Fuzzy match threshold (0-1)
    FUZZY_MATCH_THRESHOLD = 0.75

    def __init__(self, db_path: str = "tennis_gesture.db",
                 use_easyocr: bool = True):
        """
        Initialize video text extractor.

        Args:
            db_path: Path to SQLite database
            use_easyocr: Use EasyOCR (True) or fallback to simple method (False)
        """
        self.db = TennisDatabase(db_path, include_seed_data=True)
        self.use_easyocr = use_easyocr
        self._ocr_reader = None

        # Frame sampling settings
        self.frame_sample_interval = 60  # Sample every N frames

    @property
    def ocr_reader(self):
        """Lazy load OCR reader to avoid import errors if not used"""
        if self._ocr_reader is None and self.use_easyocr:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(
                    ['en'],
                    gpu=False,  # Use CPU by default for compatibility
                    verbose=False
                )
            except ImportError:
                print("Warning: EasyOCR not installed. Using fallback text extraction.")
                self.use_easyocr = False
        return self._ocr_reader

    def extract_player_name_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract player name from a video file.

        Args:
            video_path: Path to video file

        Returns:
            Identified player name or None if not found
        """
        video_path = str(video_path)

        # Check if video exists in database
        video_id = self._get_or_add_video(video_path)
        if not video_id:
            return None

        # Check if we already extracted text from this video
        existing_name = self._get_cached_player_name(video_id)
        if existing_name:
            return existing_name

        # Extract text from video frames
        ocr_results = self._extract_text_from_video(video_path, video_id)

        if not ocr_results:
            return None

        # Try to match detected text to player names
        player_name = self._match_player_name(ocr_results)

        if player_name:
            # Mark the matching OCR result
            self._mark_player_name_match(video_id, player_name)

        return player_name

    def _get_or_add_video(self, video_path: str) -> Optional[int]:
        """Get existing video ID or add video to database"""
        # Try to find existing video
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM videos WHERE file_path = ?",
                (video_path,)
            )
            row = cursor.fetchone()
            if row:
                return row['id']

        # Add video with minimal metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata = {
            'duration': total_frames / fps if fps > 0 else 0,
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames
        }

        cap.release()

        return self.db.add_video(
            filename=Path(video_path).name,
            file_path=video_path,
            metadata=metadata
        )

    def _get_cached_player_name(self, video_id: int) -> Optional[str]:
        """Check if player name was already extracted for this video"""
        with self.db.connection() as conn:
            cursor = conn.execute(
                """SELECT detected_text FROM video_text_ocr
                   WHERE video_id = ? AND is_player_name = 1
                   LIMIT 1""",
                (video_id,)
            )
            row = cursor.fetchone()
            return row['detected_text'] if row else None

    def _extract_text_from_video(self, video_path: str,
                                  video_id: int) -> List[Dict]:
        """
        Extract text from video frames using OCR.

        Processes full frames without cropping to avoid missing text
        that may appear anywhere on screen.

        Args:
            video_path: Path to video file
            video_id: Database video ID

        Returns:
            List of OCR results with text, confidence, and bounding box
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return []

        ocr_results = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames at intervals
            if frame_num % self.frame_sample_interval == 0:
                # Run OCR on full frame (no cropping assumption)
                frame_results = self._run_ocr(frame, frame_num, 0)
                ocr_results.extend(frame_results)

                # Store results in database
                self._store_ocr_results(video_id, frame_results)

            frame_num += 1

        cap.release()

        return ocr_results

    def _run_ocr(self, frame: np.ndarray, frame_num: int,
                 crop_offset_y: int) -> List[Dict]:
        """
        Run OCR on a single frame.

        Args:
            frame: Cropped frame image
            frame_num: Original frame number
            crop_offset_y: Y offset from cropping

        Returns:
            List of OCR results
        """
        results = []

        if self.use_easyocr and self.ocr_reader:
            # Use EasyOCR
            ocr_data = self.ocr_reader.readtext(frame)

            for bbox, text, confidence in ocr_data:
                # Adjust bounding box coordinates for crop offset
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] + crop_offset_y for point in bbox]

                bbox_x = min(x_coords)
                bbox_y = min(y_coords)
                bbox_w = max(x_coords) - min(x_coords)
                bbox_h = max(y_coords) - min(y_coords)

                results.append({
                    'frame_number': frame_num,
                    'detected_text': text.strip(),
                    'confidence': confidence,
                    'bbox_x': float(bbox_x),
                    'bbox_y': float(bbox_y),
                    'bbox_w': float(bbox_w),
                    'bbox_h': float(bbox_h)
                })

        return results

    def _store_ocr_results(self, video_id: int, results: List[Dict]):
        """Store OCR results in database"""
        if not results:
            return

        with self.db.connection() as conn:
            for result in results:
                conn.execute(
                    """INSERT INTO video_text_ocr
                       (video_id, frame_number, detected_text, confidence,
                        bbox_x, bbox_y, bbox_w, bbox_h)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (video_id, result['frame_number'], result['detected_text'],
                     result['confidence'], result['bbox_x'], result['bbox_y'],
                     result['bbox_w'], result['bbox_h'])
                )

    def _match_player_name(self, ocr_results: List[Dict]) -> Optional[str]:
        """
        Match OCR results to known player names.

        Args:
            ocr_results: List of OCR results

        Returns:
            Matched player name or None
        """
        # Get all professional players from database
        players = self.db.get_all_players(professionals_only=True)

        if not players:
            return None

        player_names = [p['name'] for p in players]

        best_match = None
        best_score = 0.0

        for result in ocr_results:
            text = result['detected_text']
            confidence = result['confidence']

            # Skip low confidence results
            if confidence < self.MIN_OCR_CONFIDENCE:
                continue

            # Skip text that's too short
            if len(text) < self.MIN_NAME_LENGTH:
                continue

            # Skip scoreboard patterns
            if self._is_scoreboard_text(text):
                continue

            # Skip sponsor names
            if self._is_sponsor_text(text):
                continue

            # Try to match against player names
            for player_name in player_names:
                score = self._calculate_match_score(text, player_name)

                if score > best_score and score >= self.FUZZY_MATCH_THRESHOLD:
                    best_score = score
                    best_match = player_name

        return best_match

    def _calculate_match_score(self, ocr_text: str, player_name: str) -> float:
        """
        Calculate similarity score between OCR text and player name.

        Uses multiple strategies:
        1. Exact match (normalized)
        2. Substring match
        3. Fuzzy match (SequenceMatcher)

        Args:
            ocr_text: Text from OCR
            player_name: Known player name

        Returns:
            Similarity score (0-1)
        """
        # Normalize both strings
        ocr_norm = ocr_text.lower().strip()
        player_norm = player_name.lower()

        # Exact match after normalization
        if ocr_norm == player_norm:
            return 1.0

        # Check if player name contains OCR text (or vice versa)
        if player_norm in ocr_norm or ocr_norm in player_norm:
            # Boost score for longer matches
            length_ratio = len(ocr_norm) / len(player_norm)
            if 0.5 <= length_ratio <= 1.5:
                return 0.9

        # Check last name match (often OCR captures just last name)
        ocr_parts = ocr_norm.split()
        player_parts = player_norm.split()

        if ocr_parts and player_parts:
            # Compare last names
            if ocr_parts[-1] == player_parts[-1]:
                return 0.85

            # Check if any part matches
            for ocr_part in ocr_parts:
                for player_part in player_parts:
                    if ocr_part in player_part or player_part in ocr_part:
                        if len(ocr_part) >= 3 and len(player_part) >= 3:
                            return 0.7

        # Fuzzy match using SequenceMatcher
        return SequenceMatcher(None, ocr_norm, player_norm).ratio()

    def _is_scoreboard_text(self, text: str) -> bool:
        """Check if text looks like scoreboard information"""
        import re

        # Remove common prefixes
        text_clean = text.upper().strip()

        # Check for score patterns
        if '-' in text_clean or ':' in text_clean:
            return True

        # Check for SET/GAME patterns
        if text_clean.startswith('SET') or text_clean.startswith('GAME'):
            return True

        # Check for just numbers (1-2 digits)
        if text_clean.isdigit() and len(text_clean) <= 2:
            return True

        return False

    def _is_sponsor_text(self, text: str) -> bool:
        """Check if text is a sponsor/brand name"""
        text_upper = text.upper().strip()

        for sponsor in self.SPONSOR_NAMES:
            if sponsor in text_upper or text_upper in sponsor:
                return True

        return False

    def _mark_player_name_match(self, video_id: int, player_name: str):
        """Mark the OCR result that matched a player name"""
        with self.db.connection() as conn:
            conn.execute(
                """UPDATE video_text_ocr
                   SET is_player_name = 1
                   WHERE video_id = ? AND detected_text LIKE ?""",
                (video_id, f'%{player_name.split()[-1]}%')  # Match last name
            )

    def get_detected_text(self, video_id: int,
                          include_bbox: bool = False,
                          player_names_only: bool = False) -> List[Dict]:
        """
        Get detected text from a video.

        Args:
            video_id: Database video ID
            include_bbox: Include bounding box coordinates
            player_names_only: Only return text marked as player names

        Returns:
            List of detected text results
        """
        query = "SELECT * FROM video_text_ocr WHERE video_id = ?"
        params = [video_id]

        if player_names_only:
            query += " AND is_player_name = 1"

        with self.db.connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            result = {
                'id': row['id'],
                'frame_number': row['frame_number'],
                'detected_text': row['detected_text'],
                'confidence': row['confidence'],
                'is_player_name': bool(row['is_player_name'])
            }

            if include_bbox:
                result['bbox_x'] = row['bbox_x']
                result['bbox_y'] = row['bbox_y']
                result['bbox_w'] = row['bbox_w']
                result['bbox_h'] = row['bbox_h']

            results.append(result)

        return results

    def visualize_text_detection(self, video_path: str,
                                  output_path: str = None) -> Optional[str]:
        """
        Create visualization of detected text on video frames.

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)

        Returns:
            Path to output video or None
        """
        # Get video ID
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM videos WHERE file_path = ?",
                (video_path,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        video_id = row['id']

        # Get OCR results with bounding boxes
        ocr_results = self.get_detected_text(video_id, include_bbox=True)

        if not ocr_results:
            return None

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        # Video output setup
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None

        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw bounding boxes for this frame
            frame_results = [r for r in ocr_results if r['frame_number'] == frame_num]

            for result in frame_results:
                x = int(result['bbox_x'])
                y = int(result['bbox_y'])
                w = int(result['bbox_w'])
                h = int(result['bbox_h'])

                # Color: green for player names, red for other text
                color = (0, 255, 0) if result['is_player_name'] else (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw text label
                label = result['detected_text'][:20]  # Truncate long text
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Write frame or display
            if out:
                out.write(frame)

            frame_num += 1

        cap.release()
        if out:
            out.release()

        return output_path


if __name__ == "__main__":
    # Demo usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_text_extractor.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    extractor = VideoTextExtractor()

    print(f"Extracting player name from: {video_path}")
    player_name = extractor.extract_player_name_from_video(video_path)

    if player_name:
        print(f"Identified player: {player_name}")
    else:
        print("Could not identify player from video text")

    # Show all detected text
    print("\nAll detected text:")
    with extractor.db.connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM videos WHERE file_path = ?",
            (video_path,)
        )
        row = cursor.fetchone()
        if row:
            results = extractor.get_detected_text(row['id'])
            for r in results[:20]:  # Show first 20 results
                print(f"  Frame {r['frame_number']}: {r['detected_text']} "
                      f"(confidence: {r['confidence']:.2f})")
