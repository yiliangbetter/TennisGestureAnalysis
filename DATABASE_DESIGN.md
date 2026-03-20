# Tennis Gesture Analysis - Database Design

## Overview

This document describes the database schema for storing extracted pose data from tennis videos, player information, and enabling comparison with input videos.

## Database Technology: SQLite

SQLite is recommended for this project because:
- Lightweight, serverless (single file)
- Built-in Python support via `sqlite3` module
- Supports spatial queries via extensions if needed
- Easy to backup and share

---

## Entity Relationship Diagram

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│     videos      │     │   extracted_poses   │     │  pose_landmarks     │
├─────────────────┤     ├─────────────────────┤     ├─────────────────────┤
│ id (PK)         │────<│ id (PK)             │     │ id (PK)             │
│ filename        │     │ video_id (FK)       │>────│ pose_id (FK)        │
│ file_path       │     │ frame_number        │     │ landmark_index      │
│ duration_sec    │     │ timestamp_ms        │     │ x_coord             │
│ fps             │     │ player_id (FK)      │     │ y_coord             │
│ width           │     │ stroke_type         │     │ z_coord (optional)  │
│ height          │     │ confidence_score    │     │ visibility          │
│ created_at      │     │ is_sample_pose      │     │ created_at          │
└─────────────────┘     │ created_at          │     └─────────────────────┘
                        └─────────────────────┘
                                 │
                        ┌────────┴────────┐
                        ▼                 ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│     players     │     │  gesture_sequences  │     │  joint_angles       │
├─────────────────┤     ├─────────────────────┤     ├─────────────────────┤
│ id (PK)         │────<│ id (PK)             │     │ id (PK)             │
│ name            │     │ pose_id (FK)        │     │ pose_id (FK)        │
│ country         │     │ sequence_type       │     │ angle_type          │
│ is_professional │     │ start_frame         │     │ angle_value         │
│ created_at      │     │ end_frame           │     │ calculated_at       │
└─────────────────┘     │ created_at          │     └─────────────────────┘
                        └─────────────────────┘
```

---

## Table Schemas

### 1. `players`

Stores information about tennis players (both professionals from raw_videos and users).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier |
| `name` | TEXT | NOT NULL UNIQUE | Player name (e.g., "Novak Djokovic") |
| `country` | TEXT | NULL | Country code (e.g., "SRB") |
| `is_professional` | BOOLEAN | DEFAULT 0 | True for pro players from raw_videos |
| `metadata` | TEXT | NULL | JSON string for additional info |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

### 2. `videos`

Stores metadata about processed video files.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier |
| `filename` | TEXT | NOT NULL | Original filename |
| `file_path` | TEXT | NOT NULL UNIQUE | Absolute path to video file |
| `file_hash` | TEXT | NULL | SHA256 hash for deduplication |
| `duration_sec` | REAL | NULL | Video duration in seconds |
| `fps` | REAL | NULL | Frames per second |
| `width` | INTEGER | NULL | Video width in pixels |
| `height` | INTEGER | NULL | Video height in pixels |
| `total_frames` | INTEGER | NULL | Total frame count |
| `processed` | BOOLEAN | DEFAULT 0 | Whether poses have been extracted |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

### 3. `extracted_poses`

Main table storing extracted pose data per frame.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier |
| `video_id` | INTEGER | FOREIGN KEY → videos(id) | Source video |
| `frame_number` | INTEGER | NOT NULL | Frame index (0-based) |
| `timestamp_ms` | REAL | NULL | Timestamp in milliseconds |
| `player_id` | INTEGER | FOREIGN KEY → players(id) | Detected/associated player |
| `bbox_x` | REAL | NULL | Bounding box x (normalized) |
| `bbox_y` | REAL | NULL | Bounding box y (normalized) |
| `bbox_w` | REAL | NULL | Bounding box width (normalized) |
| `bbox_h` | REAL | NULL | Bounding box height (normalized) |
| `stroke_type` | TEXT | NULL | Detected stroke type (forehand/backhand/serve) |
| `confidence_score` | REAL | DEFAULT 1.0 | Pose detection confidence |
| `is_sample_pose` | BOOLEAN | DEFAULT 0 | Flag for representative poses |
| `optical_flow_data` | BLOB | NULL | Compressed optical flow array |
| `motion_history` | BLOB | NULL | Compressed motion history image |
| `hog_features` | BLOB | NULL | Compressed HOG feature vector |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

### 4. `pose_landmarks`

Individual landmark points (33 per pose using MediaPipe format).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier |
| `pose_id` | INTEGER | FOREIGN KEY → extracted_poses(id) | Parent pose |
| `landmark_index` | INTEGER | NOT NULL | MediaPipe landmark index (0-32) |
| `x_coord` | REAL | NOT NULL | Normalized X coordinate (0-1) |
| `y_coord` | REAL | NOT NULL | Normalized Y coordinate (0-1) |
| `z_coord` | REAL | NULL | Depth coordinate (if available) |
| `visibility` | REAL | DEFAULT 1.0 | Visibility confidence (0-1) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

**Composite Index:** `UNIQUE(pose_id, landmark_index)`

### 5. `joint_angles`

Pre-calculated joint angles for faster comparison.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier |
| `pose_id` | INTEGER | FOREIGN KEY → extracted_poses(id) | Parent pose |
| `angle_type` | TEXT | NOT NULL | Type identifier (e.g., "right_elbow_flexion") |
| `angle_value` | REAL | NOT NULL | Angle in degrees (0-180) |
| `calculated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Calculation time |

**Angle Types:**
- `right_elbow_flexion` (12-14-16)
- `left_elbow_flexion` (11-13-15)
- `right_shoulder_abduction` (11-12-14)
- `left_shoulder_abduction` (12-11-13)
- `right_knee_flexion` (24-26-28)
- `left_knee_flexion` (23-25-27)
- `right_hip_angle` (23-24-26)
- `left_hip_angle` (24-23-25)
- `torso_rotation_right` (11-12-24)
- `torso_rotation_left` (12-11-23)
- `body_lean_right` (12-11-23)
- `body_lean_left` (11-12-24)

### 6. `gesture_sequences`

Groups poses into complete gesture sequences (e.g., full forehand swing).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier |
| `pose_id` | INTEGER | FOREIGN KEY → extracted_poses(id) | Key/reference pose |
| `video_id` | INTEGER | FOREIGN KEY → videos(id) | Source video |
| `player_id` | INTEGER | FOREIGN KEY → players(id) | Performing player |
| `sequence_type` | TEXT | NOT NULL | Type: forehand/backhand/serve/volley |
| `start_frame` | INTEGER | NOT NULL | Starting frame number |
| `end_frame` | INTEGER | NOT NULL | Ending frame number |
| `key_frame` | INTEGER | NULL | Most representative frame |
| `avg_confidence` | REAL | NULL | Average confidence across sequence |
| `trajectory_data` | BLOB | NULL | Compressed trajectory points |
| `velocity_profile` | BLOB | NULL | Compressed velocity vectors |
| `acceleration_profile` | BLOB | NULL | Compressed acceleration values |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

### 7. `comparison_results` (Optional - for caching)

Stores comparison results for faster repeated queries.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier |
| `input_video_path` | TEXT | NOT NULL | Path to input video |
| `input_pose_id` | INTEGER | FOREIGN KEY → extracted_poses(id) | Compared pose |
| `matched_player_id` | INTEGER | FOREIGN KEY → players(id) | Best match player |
| `matched_sequence_id` | INTEGER | FOREIGN KEY → gesture_sequences(id) | Best match sequence |
| `similarity_score` | REAL | NOT NULL | Overall similarity (0-1) |
| `pose_similarity` | REAL | NULL | Pose component score |
| `angle_similarity` | REAL | NULL | Angle component score |
| `trajectory_similarity` | REAL | NULL | Trajectory component score |
| `motion_similarity` | REAL | NULL | Motion component score |
| `recommendations` | TEXT | NULL | JSON array of recommendations |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

---

## Sample Pose Extraction Workflow

### Step 1: Process raw_videos

For each video in `raw_videos/`:
1. Extract player name from filename or use video metadata
2. Process each frame to extract:
   - 33 MediaPipe landmarks
   - Joint angles
   - Optical flow
   - Motion history
   - HOG features
3. Detect stroke type (forehand/backhand/serve)
4. Identify key frames (peak action moments)
5. Store in database

### Step 2: Create Sample Poses

For each player/stroke combination:
1. Cluster similar poses using K-Means or DBSCAN
2. Select centroid poses as "sample poses"
3. Mark with `is_sample_pose = 1`
4. Store in `gesture_sequences` table

### Step 3: Compare Input Video

For `SampleInputVideos.mp4`:
1. Extract poses using same pipeline
2. For each input pose:
   - Query `pose_landmarks` for sample poses
   - Calculate similarity scores
   - Find best match
3. Generate recommendations
4. Store results in `comparison_results`

---

## SQL Schema (CREATE statements)

```sql
-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Players table
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    country TEXT,
    is_professional BOOLEAN DEFAULT 0,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Videos table
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

-- Extracted poses table
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
    UNIQUE(video_id, frame_number)
);

-- Pose landmarks table
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

-- Joint angles table
CREATE TABLE IF NOT EXISTS joint_angles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pose_id INTEGER NOT NULL,
    angle_type TEXT NOT NULL,
    angle_value REAL NOT NULL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pose_id) REFERENCES extracted_poses(id) ON DELETE CASCADE
);

-- Gesture sequences table
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

-- Comparison results table (optional)
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_poses_video ON extracted_poses(video_id);
CREATE INDEX IF NOT EXISTS idx_poses_player ON extracted_poses(player_id);
CREATE INDEX IF NOT EXISTS idx_poses_sample ON extracted_poses(is_sample_pose);
CREATE INDEX IF NOT EXISTS idx_poses_stroke ON extracted_poses(stroke_type);
CREATE INDEX IF NOT EXISTS idx_landmarks_pose ON pose_landmarks(pose_id);
CREATE INDEX IF NOT EXISTS idx_angles_pose ON joint_angles(pose_id);
CREATE INDEX IF NOT EXISTS idx_angles_type ON joint_angles(angle_type);
CREATE INDEX IF NOT EXISTS idx_sequences_player ON gesture_sequences(player_id);
CREATE INDEX IF NOT EXISTS idx_sequences_type ON gesture_sequences(sequence_type);
CREATE INDEX IF NOT EXISTS idx_comparison_input ON comparison_results(input_video_path);
CREATE INDEX IF NOT EXISTS idx_comparison_player ON comparison_results(matched_player_id);
```

---

## Python Integration

### Database Helper Class

```python
# database_manager.py
import sqlite3
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class PoseData:
    """Represents a complete pose with all associated data"""
    id: Optional[int]
    video_id: int
    frame_number: int
    player_id: Optional[int]
    landmarks: np.ndarray  # (33, 2) array
    joint_angles: Dict[str, float]
    bbox: Optional[Tuple[float, float, float, float]]
    stroke_type: Optional[str]
    confidence: float
    optical_flow: Optional[np.ndarray]
    motion_history: Optional[np.ndarray]
    hog_features: Optional[np.ndarray]


class TennisDatabase:
    def __init__(self, db_path: str = "tennis_gesture.db"):
        self.db_path = db_path
        self._initialize_schema()

    @contextmanager
    def connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create database schema if not exists"""
        with self.connection() as conn:
            # Read and execute schema SQL
            with open("schema.sql", "r") as f:
                conn.executescript(f.read())

    def add_player(self, name: str, country: str = None,
                   is_professional: bool = True) -> int:
        """Add a player and return their ID"""
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO players (name, country, is_professional) "
                "VALUES (?, ?, ?)",
                (name, country, is_professional)
            )
            cursor = conn.execute(
                "SELECT id FROM players WHERE name = ?", (name,)
            )
            return cursor.fetchone()[0]

    def add_video(self, filename: str, file_path: str,
                  metadata: dict = None) -> int:
        """Add a video and return its ID"""
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO videos "
                "(filename, file_path, duration_sec, fps, width, height, total_frames) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (filename, file_path, metadata.get('duration'),
                 metadata.get('fps'), metadata.get('width'),
                 metadata.get('height'), metadata.get('frames'))
            )
            cursor = conn.execute(
                "SELECT id FROM videos WHERE file_path = ?", (file_path,)
            )
            return cursor.fetchone()[0]

    def add_pose(self, pose_data: PoseData) -> int:
        """Add a pose with all its landmarks and angles"""
        with self.connection() as conn:
            # Insert pose
            cursor = conn.execute(
                "INSERT INTO extracted_poses "
                "(video_id, frame_number, player_id, bbox_x, bbox_y, bbox_w, bbox_h, "
                " stroke_type, confidence_score, optical_flow_data, motion_history, hog_features) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pose_data.video_id, pose_data.frame_number, pose_data.player_id,
                 pose_data.bbox[0] if pose_data.bbox else None,
                 pose_data.bbox[1] if pose_data.bbox else None,
                 pose_data.bbox[2] if pose_data.bbox else None,
                 pose_data.bbox[3] if pose_data.bbox else None,
                 pose_data.stroke_type, pose_data.confidence,
                 pickle.dumps(pose_data.optical_flow) if pose_data.optical_flow else None,
                 pickle.dumps(pose_data.motion_history) if pose_data.motion_history else None,
                 pickle.dumps(pose_data.hog_features) if pose_data.hog_features else None)
            )
            pose_id = cursor.lastrowid

            # Insert landmarks
            for idx, (x, y) in enumerate(pose_data.landmarks):
                conn.execute(
                    "INSERT INTO pose_landmarks "
                    "(pose_id, landmark_index, x_coord, y_coord) "
                    "VALUES (?, ?, ?, ?)",
                    (pose_id, idx, float(x), float(y))
                )

            # Insert joint angles
            for angle_type, angle_value in pose_data.joint_angles.items():
                conn.execute(
                    "INSERT INTO joint_angles "
                    "(pose_id, angle_type, angle_value) "
                    "VALUES (?, ?, ?)",
                    (pose_id, angle_type, angle_value)
                )

            return pose_id

    def get_sample_poses(self, player_name: str = None,
                         stroke_type: str = None) -> List[PoseData]:
        """Retrieve sample poses for comparison"""
        query = """
            SELECT ep.id, ep.video_id, ep.frame_number, ep.player_id,
                   ep.stroke_type, ep.confidence_score
            FROM extracted_poses ep
            JOIN players p ON ep.player_id = p.id
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
            pose_data = self._load_full_pose(row[0])
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

            # Get landmarks
            cursor = conn.execute(
                "SELECT landmark_index, x_coord, y_coord, visibility "
                "FROM pose_landmarks WHERE pose_id = ? ORDER BY landmark_index",
                (pose_id,)
            )
            landmarks = np.array([(r[1], r[2]) for r in cursor.fetchall()])

            # Get joint angles
            cursor = conn.execute(
                "SELECT angle_type, angle_value FROM joint_angles WHERE pose_id = ?",
                (pose_id,)
            )
            angles = {r[0]: r[1] for r in cursor.fetchall()}

            # Get BLOB data
            optical_flow = pickle.loads(pose_row[10]) if pose_row[10] else None
            motion_history = pickle.loads(pose_row[11]) if pose_row[11] else None
            hog_features = pickle.loads(pose_row[12]) if pose_row[12] else None

            return PoseData(
                id=pose_id,
                video_id=pose_row[1],
                frame_number=pose_row[2],
                player_id=pose_row[4],
                landmarks=landmarks,
                joint_angles=angles,
                bbox=(pose_row[5], pose_row[6], pose_row[7], pose_row[8]),
                stroke_type=pose_row[9],
                confidence=pose_row[10],
                optical_flow=optical_flow,
                motion_history=motion_history,
                hog_features=hog_features
            )

    def find_similar_poses(self, input_landmarks: np.ndarray,
                           stroke_type: str = None,
                           top_k: int = 5) -> List[Tuple[PoseData, float]]:
        """Find most similar poses to input landmarks"""
        sample_poses = self.get_sample_poses(stroke_type=stroke_type)

        similarities = []
        for pose in sample_poses:
            sim = self._calculate_pose_similarity(input_landmarks, pose.landmarks)
            similarities.append((pose, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _calculate_pose_similarity(self, landmarks1: np.ndarray,
                                   landmarks2: np.ndarray) -> float:
        """Calculate similarity between two pose landmark sets"""
        if landmarks1.size == 0 or landmarks2.size == 0:
            return 0.0

        diff = landmarks1 - landmarks2
        distances = np.linalg.norm(diff, axis=1)
        avg_distance = np.mean(distances)

        max_expected_distance = 0.3
        similarity = max(0, 1 - avg_distance / max_expected_distance)
        return similarity
```

---

## Video Processing Pipeline

```python
# video_processor.py
import cv2
import os
from pathlib import Path
from database_manager import TennisDatabase, PoseData
from enhanced_gesture_analyzer import EnhancedTennisGestureAnalyzer

class VideoProcessor:
    def __init__(self, db_path: str = "tennis_gesture.db"):
        self.db = TennisDatabase(db_path)
        self.analyzer = EnhancedTennisGestureAnalyzer()

    def process_raw_videos(self, raw_videos_dir: str):
        """Process all videos in raw_videos directory"""
        video_files = list(Path(raw_videos_dir).glob("*.mp4"))

        for video_path in video_files:
            print(f"Processing: {video_path.name}")

            # Extract player name from filename
            player_name = self._extract_player_name(video_path.name)
            player_id = self.db.add_player(player_name, is_professional=True)

            # Add video to database
            video_id = self._add_video_to_db(str(video_path))

            # Process frames
            self._process_video_frames(str(video_path), video_id, player_id)

            # Mark video as processed
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE videos SET processed = 1 WHERE id = ?", (video_id,)
                )

            # Create sample poses from this video
            self._create_sample_poses(video_id, player_id)

    def _extract_player_name(self, filename: str) -> str:
        """Extract player name from video filename"""
        # Example: "tennis-djokovic-forehand-Novak Djokovic..." -> "Novak Djokovic"
        # Customize based on your naming convention
        name_mapping = {
            "djokovic": "Novak Djokovic",
            "alcaraz": "Carlos Alcaraz",
            "sinner": "Jannik Sinner",
            "zverev": "Alexander Zverev",
            "rublev": "Andrey Rublev",
            "shelton": "Ben Shelton",
        }

        filename_lower = filename.lower()
        for key, name in name_mapping.items():
            if key in filename_lower:
                return name

        # Default to filename without extension
        return Path(filename).stem

    def _add_video_to_db(self, video_path: str) -> int:
        """Add video metadata to database"""
        cap = cv2.VideoCapture(video_path)

        metadata = {
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        cap.release()

        return self.db.add_video(
            filename=os.path.basename(video_path),
            file_path=video_path,
            metadata=metadata
        )

    def _process_video_frames(self, video_path: str, video_id: int, player_id: int):
        """Process each frame and extract poses"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract landmarks
            landmarks = self.analyzer.extract_landmarks_from_frame(frame)

            if landmarks is not None:
                # Calculate joint angles
                angles = self.analyzer.calculate_joint_angles(landmarks)
                angle_dict = {
                    'right_elbow_flexion': angles[0],
                    'right_shoulder_abduction': angles[1],
                    'left_elbow_flexion': angles[2],
                    'left_shoulder_abduction': angles[3],
                    'right_knee_flexion': angles[4],
                    'right_hip_angle': angles[5],
                    'left_knee_flexion': angles[6],
                    'left_hip_angle': angles[7],
                    'torso_rotation_right': angles[8],
                    'torso_rotation_left': angles[9],
                    'body_lean_right': angles[10],
                    'body_lean_left': angles[11],
                }

                # Create pose data
                pose_data = PoseData(
                    id=None,
                    video_id=video_id,
                    frame_number=frame_num,
                    player_id=player_id,
                    landmarks=landmarks,
                    joint_angles=angle_dict,
                    bbox=None,
                    stroke_type=self._detect_stroke_type(landmarks, angle_dict),
                    confidence=1.0,
                    optical_flow=None,
                    motion_history=None,
                    hog_features=None
                )

                # Add to database
                self.db.add_pose(pose_data)

            frame_num += 1
            if frame_num % 100 == 0:
                print(f"  Processed {frame_num} frames...")

        cap.release()

    def _detect_stroke_type(self, landmarks: np.ndarray,
                            angles: dict) -> str:
        """Detect stroke type from pose"""
        # Simple heuristic-based detection
        # Can be enhanced with ML classifier

        right_elbow = angles.get('right_elbow_flexion', 0)
        torso_rotation = angles.get('torso_rotation_right', 0)

        if right_elbow > 100 and torso_rotation > 30:
            return "forehand"
        elif right_elbow < 60:
            return "serve"
        else:
            return "backhand"

    def _create_sample_poses(self, video_id: int, player_id: int):
        """Create sample poses from processed video using clustering"""
        from sklearn.cluster import KMeans

        # Get all poses for this video
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM extracted_poses WHERE video_id = ? AND player_id = ?",
                (video_id, player_id)
            )
            pose_ids = [r[0] for r in cursor.fetchall()]

        if len(pose_ids) < 3:
            return

        # Load all landmarks
        all_landmarks = []
        for pose_id in pose_ids:
            pose = self.db._load_full_pose(pose_id)
            flattened = pose.landmarks.flatten()
            all_landmarks.append(flattened)

        all_landmarks = np.array(all_landmarks)

        # Cluster poses (e.g., 5 representative poses per video)
        n_clusters = min(5, len(all_landmarks))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(all_landmarks)

        # Find closest pose to each centroid
        for i, centroid in enumerate(kmeans.cluster_centers_):
            distances = np.linalg.norm(all_landmarks - centroid, axis=1)
            closest_idx = np.argmin(distances)
            closest_pose_id = pose_ids[closest_idx]

            # Mark as sample pose
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE extracted_poses SET is_sample_pose = 1 WHERE id = ?",
                    (closest_pose_id,)
                )

                # Create gesture sequence entry
                conn.execute(
                    "INSERT INTO gesture_sequences "
                    "(pose_id, video_id, player_id, sequence_type, start_frame, end_frame, key_frame) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (closest_pose_id, video_id, player_id,
                     self._get_pose_stroke_type(closest_pose_id),
                     max(0, pose_ids.index(closest_pose_id) - 5),
                     min(len(pose_ids), pose_ids.index(closest_pose_id) + 5),
                     closest_pose_id)
                )

    def _get_pose_stroke_type(self, pose_id: int) -> str:
        """Get stroke type for a pose"""
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT stroke_type FROM extracted_poses WHERE id = ?",
                (pose_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else "forehand"

    def compare_input_video(self, input_video_path: str) -> dict:
        """Compare input video against database and return results"""
        # Process input video
        cap = cv2.VideoCapture(input_video_path)

        results = {
            'best_matches': [],
            'recommendations': []
        }

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract landmarks from input frame
            landmarks = self.analyzer.extract_landmarks_from_frame(frame)

            if landmarks is not None:
                # Find similar poses in database
                similar = self.db.find_similar_poses(landmarks, top_k=1)

                if similar:
                    matched_pose, similarity = similar[0]
                    results['best_matches'].append({
                        'frame': frame_num,
                        'matched_player_id': matched_pose.player_id,
                        'similarity': similarity
                    })

            frame_num += 1

        cap.release()

        # Calculate overall best match
        if results['best_matches']:
            # Group by player and average similarity
            player_scores = {}
            for match in results['best_matches']:
                pid = match['matched_player_id']
                if pid not in player_scores:
                    player_scores[pid] = []
                player_scores[pid].append(match['similarity'])

            best_player = max(player_scores.keys(),
                            key=lambda p: np.mean(player_scores[p]))

            results['overall_best_match'] = {
                'player_id': best_player,
                'avg_similarity': np.mean(player_scores[best_player])
            }

        return results
```

---

## Usage Example

```python
# main.py
from video_processor import VideoProcessor

# Initialize processor
processor = VideoProcessor("tennis_gesture.db")

# Process all raw videos
processor.process_raw_videos("raw_videos/")

# Compare input video
results = processor.compare_input_video("SampleInputVideos.mp4")

print(f"Best match: Player ID {results['overall_best_match']['player_id']}")
print(f"Similarity: {results['overall_best_match']['avg_similarity']:.2%}")
```

---

## Future Enhancements

1. **Vector Search**: Use SQLite with sqlite-vec extension for faster similarity search
2. **Time-series Analysis**: Add DTW (Dynamic Time Warping) for sequence comparison
3. **ML Classification**: Train a classifier for automatic stroke type detection
4. **Cloud Sync**: Optional sync to cloud database for multi-device access
5. **API Layer**: Add REST API for web/mobile app integration
