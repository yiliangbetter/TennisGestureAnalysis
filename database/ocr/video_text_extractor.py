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
    5. Return identified player name(s)

    Supports both single-player videos and compilation videos with multiple players.
    Use `extract_player_name_from_video()` for single best match, or
    `extract_all_player_names_from_video()` to get all detected players.
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
        Returns the highest-confidence match (for single-player videos).

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

    def extract_all_player_names_from_video(self, video_path: str) -> List[str]:
        """
        Extract all player names from a video file.
        Useful for compilation videos containing multiple players.

        Args:
            video_path: Path to video file

        Returns:
            List of identified player names (may be empty if none found)
        """
        video_path = str(video_path)

        # Check if video exists in database
        video_id = self._get_or_add_video(video_path)
        if not video_id:
            return []

        # Extract text from video frames (may already be cached)
        ocr_results = self._extract_text_from_video(video_path, video_id)

        if not ocr_results:
            return []

        # Match all detected player names
        all_players = self._match_all_player_names(ocr_results)

        # Mark all matched players in database
        if all_players:
            for player_name in all_players:
                self._mark_player_name_match(video_id, player_name)

        return all_players

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
        Returns the first (highest confidence) match for backward compatibility.

        Args:
            ocr_results: List of OCR results

        Returns:
            Matched player name or None
        """
        all_matches = self._match_all_player_names(ocr_results)
        return all_matches[0] if all_matches else None

    def _match_all_player_names(self, ocr_results: List[Dict]) -> List[str]:
        """
        Match OCR results to all known player names.

        For compilation videos that contain multiple players, this method
        returns all detected players sorted by confidence (highest first).

        Matching criteria:
        - OCR confidence must be >= MIN_OCR_CONFIDENCE
        - Text length must be >= MIN_NAME_LENGTH
        - Fuzzy match score must be >= FUZZY_MATCH_THRESHOLD
        - Combined score (OCR_conf * match_score) must be >= 0.4
        - Player must be detected in at least 1 frame (tracked via detection_count)

        Args:
            ocr_results: List of OCR results

        Returns:
            List of matched player names, sorted by combined score (descending)
        """
        # Get all professional players from database
        players = self.db.get_all_players(professionals_only=True)

        if not players:
            return []

        player_names = [p['name'] for p in players]

        # Track best combined score and detection count for each player
        # combined_score = max(ocr_confidence * match_score) across all detections
        player_scores: Dict[str, float] = {}
        player_detection_count: Dict[str, int] = {}

        for result in ocr_results:
            text = result['detected_text']
            ocr_confidence = result['confidence']

            # Skip low confidence OCR results
            if ocr_confidence < self.MIN_OCR_CONFIDENCE:
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

            # Track if this text matched any known player
            matched_known_player = False

            # Try to match against player names
            for player_name in player_names:
                match_score = self._calculate_match_score(text, player_name)

                if match_score >= self.FUZZY_MATCH_THRESHOLD:
                    matched_known_player = True

                    # Combined score factors in both OCR confidence and match quality
                    combined_score = ocr_confidence * match_score

                    # Require minimum combined score of 0.4
                    if combined_score < 0.4:
                        continue

                    # Track detection count (number of independent detections)
                    if player_name not in player_detection_count:
                        player_detection_count[player_name] = 0
                    player_detection_count[player_name] += 1

                    # Keep the highest combined score for each player
                    if player_name not in player_scores or combined_score > player_scores[player_name]:
                        player_scores[player_name] = combined_score

            # If no known player matched and this looks like a name, add as candidate
            if not matched_known_player and self._looks_like_player_name(text, ocr_confidence):
                self._add_player_candidate(text, ocr_confidence)

        # Filter to players with at least 1 detection and sort by score
        sorted_players = sorted(player_scores.items(), key=lambda x: (-x[1], x[0]))
        return [player for player, score in sorted_players]

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

    def _looks_like_player_name(self, text: str, confidence: float) -> bool:
        """
        Check if the detected text looks like it could be a player's name.

        Heuristics used:
        - Must have at least 2 words (first and last name)
        - Must have reasonable length (4-30 chars)
        - OCR confidence should be reasonably high (>= 0.6)

        Args:
            text: Detected text from OCR
            confidence: OCR confidence score

        Returns:
            True if text looks like a potential player name
        """
        # Basic confidence threshold
        if confidence < 0.6:
            return False

        # Length check
        if len(text) < 4 or len(text) > 30:
            return False

        words = text.split()

        # Must have at least 2 words (first and last name)
        if len(words) < 2:
            return False

        # Check against sponsor/scoreboard patterns to avoid false positives
        if self._is_sponsor_text(text):
            return False

        if self._is_scoreboard_text(text):
            return False

        return True

    def _add_player_candidate(self, name: str, confidence: float) -> Optional[int]:
        """
        Add a potential new player to the database as a candidate.

        Candidates are marked with is_professional=0 (or use metadata to flag as candidate)
        and can be reviewed manually to confirm if they are real players.

        Args:
            name: Detected player name
            confidence: OCR confidence score

        Returns:
            Player ID if added, None if already exists or failed
        """
        # Normalize the name
        name = name.strip()
        if not name:
            return None

        with self.db.connection() as conn:
            # Check if this name already exists (case-insensitive)
            cursor = conn.execute(
                "SELECT id FROM players WHERE LOWER(name) = LOWER(?)",
                (name,)
            )
            if cursor.fetchone():
                # Already exists
                return None

            try:
                # Insert as candidate player
                # is_professional=0 and metadata flags it as OCR candidate
                cursor = conn.execute(
                    """INSERT INTO players (name, country, is_professional, metadata, created_at)
                       VALUES (?, 'UNK', 0, ?, datetime('now'))""",
                    (name, f'{{"source": "OCR", "confidence": {confidence:.2f}, "status": "candidate"}}')
                )
                conn.commit()

                print(f"    Added new player candidate: {name} (OCR confidence: {confidence:.2f})")
                return cursor.lastrowid

            except Exception as e:
                print(f"    Failed to add player candidate '{name}': {e}")
                return None

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

    print(f"Extracting player names from: {video_path}")
    print("=" * 60)

    # Extract all player names (supports compilation videos)
    all_players = extractor.extract_all_player_names_from_video(video_path)

    if all_players:
        print(f"\nDetected {len(all_players)} player(s):")
        for i, player_name in enumerate(all_players, 1):
            print(f"  {i}. {player_name}")
    else:
        print("Could not identify any players from video text")

    # Show all detected text
    print("\nAll detected text (first 30):")
    with extractor.db.connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM videos WHERE file_path = ?",
            (video_path,)
        )
        row = cursor.fetchone()
        if row:
            results = extractor.get_detected_text(row['id'])
            for r in results[:30]:
                marker = " [PLAYER]" if r['is_player_name'] else ""
                print(f"  Frame {r['frame_number']}: {r['detected_text']} "
                      f"(confidence: {r['confidence']:.2f}){marker}")
