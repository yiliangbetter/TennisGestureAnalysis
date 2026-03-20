#!/usr/bin/env python3
"""
Database Manager for Tennis Gesture Analysis

Provides SQLite database operations for storing and querying pose data,
player information, and gesture comparisons.
"""

import sqlite3
import hashlib
import pickle
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import json


@dataclass
class PoseData:
    """Represents a complete pose with all associated data"""
    id: Optional[int]
    video_id: int
    frame_number: int
    player_id: Optional[int]
    landmarks: np.ndarray  # (33, 2) array of (x, y) coordinates
    joint_angles: Dict[str, float] = field(default_factory=dict)
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h)
    stroke_type: Optional[str] = None
    confidence: float = 1.0
    optical_flow: Optional[np.ndarray] = None
    motion_history: Optional[np.ndarray] = None
    hog_features: Optional[np.ndarray] = None
    timestamp_ms: Optional[float] = None
    is_sample_pose: bool = False


class TennisDatabase:
    """
    SQLite database manager for tennis gesture analysis.

    Handles storage and retrieval of:
    - Player information
    - Video metadata
    - Extracted pose landmarks
    - Joint angles
    - Gesture sequences
    - Comparison results
    """

    # MediaPipe landmark indices for reference
    LANDMARK_NAMES = {
        0: 'nose',
        1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
        4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
        7: 'left_ear', 8: 'right_ear',
        9: 'mouth_left', 10: 'mouth_right',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        17: 'left_pinky', 18: 'right_pinky',
        19: 'left_index', 20: 'right_index',
        21: 'left_thumb', 22: 'right_thumb',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle',
        29: 'left_heel', 30: 'right_heel',
        31: 'left_foot_index', 32: 'right_foot_index',
    }

    def __init__(self, db_path: str = "tennis_gesture.db"):
        """
        Initialize database connection and create schema if needed.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_schema()

    @contextmanager
    def connection(self):
        """Context manager for database connections with transaction handling"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_schema(self, include_seed_data: bool = False):
        """
        Create database schema if it doesn't exist.

        Args:
            include_seed_data: If True and schema.sql exists, include seed data
                              (default: False for clean test databases)
        """
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            # Inline schema if file doesn't exist
            self._create_schema_inline()
            return

        with self.connection() as conn:
            with open(schema_path, "r") as f:
                schema_content = f.read()

            if not include_seed_data:
                # Remove INSERT statements (seed data) to avoid populating
                # test databases with default data
                cleaned_schema = re.sub(
                    r'^\s*INSERT\s+[^;]+;\s*$',
                    '',
                    schema_content,
                    flags=re.IGNORECASE | re.MULTILINE
                )
            else:
                cleaned_schema = schema_content

            conn.executescript(cleaned_schema)

    def _create_schema_inline(self):
        """Create schema inline if schema.sql doesn't exist"""
        with self.connection() as conn:
            conn.executescript("""
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    country TEXT,
                    is_professional BOOLEAN DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    file_hash TEXT,
                    duration_sec REAL,
                    fps REAL,
                    width INTEGER,
                    height INTEGER,
                    total_frames INTEGER,
                    processed BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS extracted_poses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    frame_number INTEGER NOT NULL,
                    timestamp_ms REAL,
                    player_id INTEGER,
                    bbox_x REAL,
                    bbox_y REAL,
                    bbox_w REAL,
                    bbox_h REAL,
                    stroke_type TEXT,
                    confidence_score REAL DEFAULT 1.0,
                    is_sample_pose BOOLEAN DEFAULT 0,
                    optical_flow_data BLOB,
                    motion_history BLOB,
                    hog_features BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE SET NULL,
                    UNIQUE(video_id, frame_number, player_id)
                );

                CREATE TABLE IF NOT EXISTS pose_landmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pose_id INTEGER NOT NULL,
                    landmark_index INTEGER NOT NULL,
                    x_coord REAL NOT NULL,
                    y_coord REAL NOT NULL,
                    z_coord REAL,
                    visibility REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pose_id) REFERENCES extracted_poses(id) ON DELETE CASCADE,
                    UNIQUE(pose_id, landmark_index)
                );

                CREATE TABLE IF NOT EXISTS joint_angles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pose_id INTEGER NOT NULL,
                    angle_type TEXT NOT NULL,
                    angle_value REAL NOT NULL,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pose_id) REFERENCES extracted_poses(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS gesture_sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pose_id INTEGER,
                    video_id INTEGER NOT NULL,
                    player_id INTEGER NOT NULL,
                    sequence_type TEXT NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    key_frame INTEGER,
                    avg_confidence REAL,
                    trajectory_data BLOB,
                    velocity_profile BLOB,
                    acceleration_profile BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pose_id) REFERENCES extracted_poses(id) ON DELETE SET NULL,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS comparison_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_video_path TEXT NOT NULL,
                    input_pose_id INTEGER,
                    matched_player_id INTEGER,
                    matched_sequence_id INTEGER,
                    similarity_score REAL NOT NULL,
                    pose_similarity REAL,
                    angle_similarity REAL,
                    trajectory_similarity REAL,
                    motion_similarity REAL,
                    recommendations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (input_pose_id) REFERENCES extracted_poses(id) ON DELETE SET NULL,
                    FOREIGN KEY (matched_player_id) REFERENCES players(id) ON DELETE SET NULL,
                    FOREIGN KEY (matched_sequence_id) REFERENCES gesture_sequences(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_poses_video ON extracted_poses(video_id);
                CREATE INDEX IF NOT EXISTS idx_poses_player ON extracted_poses(player_id);
                CREATE INDEX IF NOT EXISTS idx_poses_sample ON extracted_poses(is_sample_pose);
                CREATE INDEX IF NOT EXISTS idx_poses_stroke ON extracted_poses(stroke_type);
                CREATE INDEX IF NOT EXISTS idx_landmarks_pose ON pose_landmarks(pose_id);
                CREATE INDEX IF NOT EXISTS idx_angles_pose ON joint_angles(pose_id);
                CREATE INDEX IF NOT EXISTS idx_sequences_player ON gesture_sequences(player_id);
            """)

    # =========================================================================
    # PLAYER OPERATIONS
    # =========================================================================

    def add_player(self, name: str, country: str = None,
                   is_professional: bool = True, metadata: dict = None) -> int:
        """
        Add a player to the database.

        Args:
            name: Player name (must be unique)
            country: Country code (e.g., "SRB", "ESP")
            is_professional: True for professional players
            metadata: Additional info as dict (stored as JSON)

        Returns:
            Player ID
        """
        with self.connection() as conn:
            # Try to get existing player first
            cursor = conn.execute(
                "SELECT id FROM players WHERE name = ?", (name,)
            )
            row = cursor.fetchone()
            if row:
                return row['id']

            # Insert new player
            metadata_json = json.dumps(metadata) if metadata else None
            conn.execute(
                "INSERT INTO players (name, country, is_professional, metadata) "
                "VALUES (?, ?, ?, ?)",
                (name, country, is_professional, metadata_json)
            )

            # Get the new ID
            cursor = conn.execute(
                "SELECT id FROM players WHERE name = ?", (name,)
            )
            return cursor.fetchone()['id']

    def get_player(self, player_id: int) -> Optional[dict]:
        """Get player by ID"""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM players WHERE id = ?", (player_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_player_by_name(self, name: str) -> Optional[dict]:
        """Get player by name"""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM players WHERE name = ?", (name,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_players(self, professionals_only: bool = False) -> List[dict]:
        """Get all players"""
        query = "SELECT * FROM players"
        params = ()
        if professionals_only:
            query += " WHERE is_professional = 1"

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # VIDEO OPERATIONS
    # =========================================================================

    def add_video(self, filename: str, file_path: str,
                  metadata: dict = None, file_hash: str = None) -> int:
        """
        Add a video to the database.

        Args:
            filename: Original filename
            file_path: Absolute path to video file
            metadata: Dict with duration, fps, width, height, total_frames
            file_hash: Optional pre-computed SHA256 hash of the file.
                       If not provided, hash is calculated only if file exists.

        Returns:
            Video ID
        """
        with self.connection() as conn:
            # Check if video already exists
            cursor = conn.execute(
                "SELECT id FROM videos WHERE file_path = ?", (file_path,)
            )
            row = cursor.fetchone()
            if row:
                return row['id']

            # Calculate file hash for deduplication (only if file exists or hash provided)
            if file_hash is None:
                file_hash = self._calculate_file_hash(file_path)

            # Extract metadata
            duration = metadata.get('duration') if metadata else None
            fps = metadata.get('fps') if metadata else None
            width = metadata.get('width') if metadata else None
            height = metadata.get('height') if metadata else None
            total_frames = metadata.get('total_frames') if metadata else None

            conn.execute(
                "INSERT INTO videos "
                "(filename, file_path, file_hash, duration_sec, fps, width, height, total_frames) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (filename, file_path, file_hash, duration, fps, width, height, total_frames)
            )

            # Get the new ID
            cursor = conn.execute(
                "SELECT id FROM videos WHERE file_path = ?", (file_path,)
            )
            return cursor.fetchone()['id']

    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate SHA256 hash of file.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hex digest, or None if file doesn't exist
        """
        if not Path(file_path).exists():
            return None

        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_video(self, video_id: int) -> Optional[dict]:
        """Get video by ID"""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM videos WHERE id = ?", (video_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def mark_video_processed(self, video_id: int):
        """Mark video as processed"""
        with self.connection() as conn:
            conn.execute(
                "UPDATE videos SET processed = 1 WHERE id = ?", (video_id,)
            )

    # =========================================================================
    # POSE OPERATIONS
    # =========================================================================

    def add_pose(self, pose_data: PoseData) -> int:
        """
        Add a pose with all its landmarks and angles.

        Args:
            pose_data: PoseData object with complete pose information

        Returns:
            Pose ID
        """
        with self.connection() as conn:
            # Serialize BLOB data
            optical_flow_blob = pickle.dumps(pose_data.optical_flow) if pose_data.optical_flow is not None else None
            motion_history_blob = pickle.dumps(pose_data.motion_history) if pose_data.motion_history is not None else None
            hog_features_blob = pickle.dumps(pose_data.hog_features) if pose_data.hog_features is not None else None

            # Insert pose
            cursor = conn.execute(
                "INSERT INTO extracted_poses "
                "(video_id, frame_number, player_id, timestamp_ms, "
                " bbox_x, bbox_y, bbox_w, bbox_h, "
                " stroke_type, confidence_score, is_sample_pose, "
                " optical_flow_data, motion_history, hog_features) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pose_data.video_id, pose_data.frame_number, pose_data.player_id,
                 pose_data.timestamp_ms,
                 pose_data.bbox[0] if pose_data.bbox else None,
                 pose_data.bbox[1] if pose_data.bbox else None,
                 pose_data.bbox[2] if pose_data.bbox else None,
                 pose_data.bbox[3] if pose_data.bbox else None,
                 pose_data.stroke_type, pose_data.confidence,
                 1 if pose_data.is_sample_pose else 0,
                 optical_flow_blob, motion_history_blob, hog_features_blob)
            )
            pose_id = cursor.lastrowid

            # Insert landmarks (batch insert for efficiency)
            landmark_data = []
            for idx, (x, y) in enumerate(pose_data.landmarks):
                landmark_data.append((pose_id, idx, float(x), float(y)))

            conn.executemany(
                "INSERT INTO pose_landmarks (pose_id, landmark_index, x_coord, y_coord) "
                "VALUES (?, ?, ?, ?)",
                landmark_data
            )

            # Insert joint angles
            if pose_data.joint_angles:
                angle_data = [
                    (pose_id, angle_type, angle_value)
                    for angle_type, angle_value in pose_data.joint_angles.items()
                ]
                conn.executemany(
                    "INSERT INTO joint_angles (pose_id, angle_type, angle_value) "
                    "VALUES (?, ?, ?)",
                    angle_data
                )

            return pose_id

    def get_pose(self, pose_id: int) -> Optional[PoseData]:
        """Get complete pose data by ID"""
        return self._load_full_pose(pose_id)

    def get_sample_poses(self, player_name: str = None,
                         stroke_type: str = None) -> List[PoseData]:
        """
        Retrieve sample poses for comparison.

        Args:
            player_name: Filter by player name (optional)
            stroke_type: Filter by stroke type (optional)

        Returns:
            List of PoseData objects
        """
        query = """
            SELECT ep.id, ep.video_id, ep.frame_number, ep.player_id,
                   ep.stroke_type, ep.confidence_score, ep.timestamp_ms,
                   ep.bbox_x, ep.bbox_y, ep.bbox_w, ep.bbox_h,
                   p.name AS player_name
            FROM extracted_poses ep
            LEFT JOIN players p ON ep.player_id = p.id
            WHERE ep.is_sample_pose = 1
        """
        params = []

        if player_name:
            query += " AND p.name = ?"
            params.append(player_name)

        if stroke_type:
            query += " AND ep.stroke_type = ?"
            params.append(stroke_type)

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        poses = []
        for row in rows:
            pose_data = self._load_full_pose(row['id'])
            poses.append(pose_data)

        return poses

    def _load_full_pose(self, pose_id: int) -> PoseData:
        """Load complete pose data including landmarks and angles"""
        with self.connection() as conn:
            # Get base pose data
            cursor = conn.execute(
                "SELECT * FROM extracted_poses WHERE id = ?", (pose_id,)
            )
            pose_row = cursor.fetchone()

            if not pose_row:
                return None

            # Get landmarks
            cursor = conn.execute(
                "SELECT landmark_index, x_coord, y_coord, visibility "
                "FROM pose_landmarks WHERE pose_id = ? ORDER BY landmark_index",
                (pose_id,)
            )
            landmarks = np.array([(r['x_coord'], r['y_coord']) for r in cursor.fetchall()])

            # Get joint angles
            cursor = conn.execute(
                "SELECT angle_type, angle_value FROM joint_angles WHERE pose_id = ?",
                (pose_id,)
            )
            angles = {r['angle_type']: r['angle_value'] for r in cursor.fetchall()}

            # Deserialize BLOB data
            optical_flow = pickle.loads(pose_row['optical_flow_data']) if pose_row['optical_flow_data'] else None
            motion_history = pickle.loads(pose_row['motion_history']) if pose_row['motion_history'] else None
            hog_features = pickle.loads(pose_row['hog_features']) if pose_row['hog_features'] else None

            return PoseData(
                id=pose_id,
                video_id=pose_row['video_id'],
                frame_number=pose_row['frame_number'],
                player_id=pose_row['player_id'],
                landmarks=landmarks,
                joint_angles=angles,
                bbox=(pose_row['bbox_x'], pose_row['bbox_y'],
                      pose_row['bbox_w'], pose_row['bbox_h']),
                stroke_type=pose_row['stroke_type'],
                confidence=pose_row['confidence_score'],
                optical_flow=optical_flow,
                motion_history=motion_history,
                hog_features=hog_features,
                timestamp_ms=pose_row['timestamp_ms']
            )

    def get_poses_by_video(self, video_id: int) -> List[PoseData]:
        """Get all poses from a video"""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM extracted_poses WHERE video_id = ? ORDER BY frame_number",
                (video_id,)
            )
            pose_ids = [r['id'] for r in cursor.fetchall()]

        return [self._load_full_pose(pid) for pid in pose_ids if pid]

    def get_poses_by_player(self, player_id: int,
                            stroke_type: str = None) -> List[PoseData]:
        """Get all poses for a player"""
        query = "SELECT id FROM extracted_poses WHERE player_id = ?"
        params = [player_id]

        if stroke_type:
            query += " AND stroke_type = ?"
            params.append(stroke_type)

        query += " ORDER BY frame_number"

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            pose_ids = [r['id'] for r in cursor.fetchall()]

        return [self._load_full_pose(pid) for pid in pose_ids if pid]

    # =========================================================================
    # SIMILARITY SEARCH
    # =========================================================================

    def find_similar_poses(self, input_landmarks: np.ndarray,
                           stroke_type: str = None,
                           top_k: int = 5) -> List[Tuple[PoseData, float]]:
        """
        Find most similar poses to input landmarks.

        Args:
            input_landmarks: (33, 2) array of normalized landmarks
            stroke_type: Filter by stroke type (optional)
            top_k: Number of results to return

        Returns:
            List of (PoseData, similarity_score) tuples
        """
        sample_poses = self.get_sample_poses(stroke_type=stroke_type)

        similarities = []
        for pose in sample_poses:
            sim = self.calculate_pose_similarity(input_landmarks, pose.landmarks)
            similarities.append((pose, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def calculate_pose_similarity(self, landmarks1: np.ndarray,
                                  landmarks2: np.ndarray) -> float:
        """
        Calculate similarity between two pose landmark sets.

        Uses normalized Euclidean distance converted to 0-1 similarity.

        Args:
            landmarks1: First pose landmarks (33, 2)
            landmarks2: Second pose landmarks (33, 2)

        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        if landmarks1.size == 0 or landmarks2.size == 0:
            return 0.0

        diff = landmarks1 - landmarks2
        distances = np.linalg.norm(diff, axis=1)
        avg_distance = np.mean(distances)

        # Convert to similarity using sigmoid-like function
        # This ensures similarity is always > 0 for finite distances
        # Tuned for tennis movements: small distances yield high similarity
        similarity = 1.0 / (1.0 + avg_distance)
        return similarity

    def calculate_angle_similarity(self, angles1: dict, angles2: dict) -> float:
        """Calculate similarity between joint angle sets"""
        if not angles1 or not angles2:
            return 0.0

        # Find common angle types
        common_types = set(angles1.keys()) & set(angles2.keys())
        if not common_types:
            return 0.0

        diffs = [abs(angles1[t] - angles2[t]) for t in common_types]
        avg_diff = sum(diffs) / len(diffs)

        # Convert to similarity (up to 30 degrees difference is acceptable)
        max_expected_diff = 30.0
        similarity = max(0, 1 - avg_diff / max_expected_diff)
        return similarity

    # =========================================================================
    # GESTURE SEQUENCE OPERATIONS
    # =========================================================================

    def add_gesture_sequence(self, pose_id: int, video_id: int, player_id: int,
                             sequence_type: str, start_frame: int,
                             end_frame: int, key_frame: int = None,
                             trajectory_data: np.ndarray = None,
                             velocity_profile: np.ndarray = None,
                             acceleration_profile: np.ndarray = None) -> int:
        """Add a gesture sequence"""
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO gesture_sequences "
                "(pose_id, video_id, player_id, sequence_type, "
                " start_frame, end_frame, key_frame, "
                " trajectory_data, velocity_profile, acceleration_profile) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pose_id, video_id, player_id, sequence_type,
                 start_frame, end_frame, key_frame,
                 pickle.dumps(trajectory_data) if trajectory_data is not None else None,
                 pickle.dumps(velocity_profile) if velocity_profile is not None else None,
                 pickle.dumps(acceleration_profile) if acceleration_profile is not None else None)
            )
            return cursor.lastrowid

    def get_sequences_by_player(self, player_id: int,
                                sequence_type: str = None) -> List[dict]:
        """Get gesture sequences for a player"""
        query = "SELECT * FROM gesture_sequences WHERE player_id = ?"
        params = [player_id]

        if sequence_type:
            query += " AND sequence_type = ?"
            params.append(sequence_type)

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            sequences = []
            for row in cursor.fetchall():
                seq = dict(row)
                # Deserialize BLOBs
                if seq['trajectory_data']:
                    seq['trajectory_data'] = pickle.loads(seq['trajectory_data'])
                if seq['velocity_profile']:
                    seq['velocity_profile'] = pickle.loads(seq['velocity_profile'])
                if seq['acceleration_profile']:
                    seq['acceleration_profile'] = pickle.loads(seq['acceleration_profile'])
                sequences.append(seq)
            return sequences

    # =========================================================================
    # COMPARISON RESULTS
    # =========================================================================

    def save_comparison_result(self, input_video_path: str,
                               matched_player_id: int,
                               similarity_score: float,
                               input_pose_id: int = None,
                               matched_sequence_id: int = None,
                               pose_similarity: float = None,
                               angle_similarity: float = None,
                               trajectory_similarity: float = None,
                               motion_similarity: float = None,
                               recommendations: List[str] = None) -> int:
        """Save a comparison result"""
        with self.connection() as conn:
            recs_json = json.dumps(recommendations) if recommendations else None

            cursor = conn.execute(
                "INSERT INTO comparison_results "
                "(input_video_path, input_pose_id, matched_player_id, "
                " matched_sequence_id, similarity_score, "
                " pose_similarity, angle_similarity, "
                " trajectory_similarity, motion_similarity, recommendations) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (input_video_path, input_pose_id, matched_player_id,
                 matched_sequence_id, similarity_score,
                 pose_similarity, angle_similarity,
                 trajectory_similarity, motion_similarity, recs_json)
            )
            return cursor.lastrowid

    def get_comparison_history(self, input_video_path: str = None) -> List[dict]:
        """Get comparison results, optionally filtered by input video"""
        query = "SELECT * FROM comparison_results"
        params = ()

        if input_video_path:
            query += " WHERE input_video_path = ?"
            params = (input_video_path,)

        query += " ORDER BY created_at DESC"

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_statistics(self) -> dict:
        """Get database statistics"""
        with self.connection() as conn:
            stats = {}

            # Count players
            cursor = conn.execute("SELECT COUNT(*) FROM players")
            stats['total_players'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM players WHERE is_professional = 1")
            stats['professional_players'] = cursor.fetchone()[0]

            # Count videos
            cursor = conn.execute("SELECT COUNT(*) FROM videos")
            stats['total_videos'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM videos WHERE processed = 1")
            stats['processed_videos'] = cursor.fetchone()[0]

            # Count poses
            cursor = conn.execute("SELECT COUNT(*) FROM extracted_poses")
            stats['total_poses'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM extracted_poses WHERE is_sample_pose = 1")
            stats['sample_poses'] = cursor.fetchone()[0]

            # Count by stroke type
            cursor = conn.execute(
                "SELECT stroke_type, COUNT(*) FROM extracted_poses GROUP BY stroke_type"
            )
            stats['poses_by_stroke'] = {r['stroke_type']: r[1] for r in cursor.fetchall() if r['stroke_type']}

            # Count landmarks
            cursor = conn.execute("SELECT COUNT(*) FROM pose_landmarks")
            stats['total_landmarks'] = cursor.fetchone()[0]

            # Count angles
            cursor = conn.execute("SELECT COUNT(*) FROM joint_angles")
            stats['total_angles'] = cursor.fetchone()[0]

            return stats

    def close(self):
        """Close database connection (cleanup)"""
        # Connection is managed via context manager, but this ensures cleanup
        pass


if __name__ == "__main__":
    # Demo usage
    db = TennisDatabase("test_tennis.db")

    # Add players
    djokovic_id = db.add_player("Novak Djokovic", country="SRB")
    alcaraz_id = db.add_player("Carlos Alcaraz", country="ESP")

    print(f"Added players: Djokovic={djokovic_id}, Alcaraz={alcaraz_id}")

    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nDatabase test complete!")
