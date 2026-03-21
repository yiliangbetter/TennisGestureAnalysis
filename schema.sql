-- Tennis Gesture Analysis Database Schema
-- SQLite compatible

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- ============================================================================
-- PLAYERS TABLE
-- Stores information about tennis players (professionals and users)
-- ============================================================================
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    country TEXT,
    is_professional BOOLEAN DEFAULT 0,
    metadata TEXT,  -- JSON string for additional info
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- VIDEOS TABLE
-- Stores metadata about processed video files
-- ============================================================================
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    file_hash TEXT,  -- SHA256 for deduplication
    duration_sec REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    total_frames INTEGER,
    processed BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- EXTRACTED_POSES TABLE
-- Main table storing extracted pose data per frame
-- ============================================================================
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
    stroke_type TEXT,  -- forehand, backhand, serve, volley
    confidence_score REAL DEFAULT 1.0,
    is_sample_pose BOOLEAN DEFAULT 0,  -- Flag for representative poses
    optical_flow_data BLOB,  -- Compressed optical flow array
    motion_history BLOB,     -- Compressed motion history image
    hog_features BLOB,       -- Compressed HOG feature vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE SET NULL,
    UNIQUE(video_id, frame_number, player_id)
);

-- ============================================================================
-- POSE_LANDMARKS TABLE
-- Individual landmark points (33 per pose using MediaPipe format)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pose_landmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pose_id INTEGER NOT NULL,
    landmark_index INTEGER NOT NULL,  -- 0-32 MediaPipe indices
    x_coord REAL NOT NULL,            -- Normalized X (0-1)
    y_coord REAL NOT NULL,            -- Normalized Y (0-1)
    z_coord REAL,                     -- Depth (if available)
    visibility REAL DEFAULT 1.0,      -- Visibility confidence
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pose_id) REFERENCES extracted_poses(id) ON DELETE CASCADE,
    UNIQUE(pose_id, landmark_index)
);

-- ============================================================================
-- JOINT_ANGLES TABLE
-- Pre-calculated joint angles for faster comparison
-- ============================================================================
CREATE TABLE IF NOT EXISTS joint_angles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pose_id INTEGER NOT NULL,
    angle_type TEXT NOT NULL,
    angle_value REAL NOT NULL,        -- Angle in degrees
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pose_id) REFERENCES extracted_poses(id) ON DELETE CASCADE
);

-- ============================================================================
-- GESTURE_SEQUENCES TABLE
-- Groups poses into complete gesture sequences
-- ============================================================================
CREATE TABLE IF NOT EXISTS gesture_sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pose_id INTEGER,
    video_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    sequence_type TEXT NOT NULL,      -- forehand/backhand/serve/volley
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    key_frame INTEGER,                -- Most representative frame
    avg_confidence REAL,
    trajectory_data BLOB,             -- Compressed trajectory points
    velocity_profile BLOB,            -- Compressed velocity vectors
    acceleration_profile BLOB,        -- Compressed acceleration values
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pose_id) REFERENCES extracted_poses(id) ON DELETE SET NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
);

-- ============================================================================
-- COMPARISON_RESULTS TABLE (Optional - for caching)
-- Stores comparison results for faster repeated queries
-- ============================================================================
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
    recommendations TEXT,             -- JSON array of recommendations
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (input_pose_id) REFERENCES extracted_poses(id) ON DELETE SET NULL,
    FOREIGN KEY (matched_player_id) REFERENCES players(id) ON DELETE SET NULL,
    FOREIGN KEY (matched_sequence_id) REFERENCES gesture_sequences(id) ON DELETE SET NULL
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================
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

-- ============================================================================
-- VIDEO TEXT OCR TABLE
-- Stores text extracted from videos using OCR for player name identification
-- ============================================================================
CREATE TABLE IF NOT EXISTS video_text_ocr (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    frame_number INTEGER,
    detected_text TEXT NOT NULL,
    confidence REAL,
    bbox_x REAL,
    bbox_y REAL,
    bbox_w REAL,
    bbox_h REAL,
    is_player_name BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ocr_video ON video_text_ocr(video_id);
CREATE INDEX IF NOT EXISTS idx_ocr_is_player_name ON video_text_ocr(is_player_name);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Complete pose with player info
CREATE VIEW IF NOT EXISTS pose_summary AS
SELECT
    ep.id AS pose_id,
    ep.frame_number,
    ep.stroke_type,
    ep.confidence_score,
    ep.is_sample_pose,
    p.name AS player_name,
    p.is_professional,
    v.filename AS video_filename,
    ep.created_at
FROM extracted_poses ep
LEFT JOIN players p ON ep.player_id = p.id
LEFT JOIN videos v ON ep.video_id = v.id;

-- View: Sample poses ready for comparison
CREATE VIEW IF NOT EXISTS sample_poses_view AS
SELECT
    ep.id AS pose_id,
    p.name AS player_name,
    ep.stroke_type,
    ep.frame_number,
    v.filename AS video_filename
FROM extracted_poses ep
JOIN players p ON ep.player_id = p.id
JOIN videos v ON ep.video_id = v.id
WHERE ep.is_sample_pose = 1
ORDER BY p.name, ep.stroke_type;

-- ============================================================================
-- TRIGGERS FOR DATA INTEGRITY
-- ============================================================================

-- Trigger: Auto-update video processed flag when poses are added
CREATE TRIGGER IF NOT EXISTS update_video_processed
AFTER INSERT ON extracted_poses
BEGIN
    UPDATE videos SET processed = 1 WHERE id = NEW.video_id;
END;

-- ============================================================================
-- SEED DATA (Optional)
-- ============================================================================

-- ============================================================================
-- SEED DATA: Professional Tennis Players (50 players)
-- ============================================================================
INSERT OR IGNORE INTO players (name, is_professional) VALUES
    -- Men: Big 4 + Legends
    ('Novak Djokovic', 1),
    ('Rafael Nadal', 1),
    ('Roger Federer', 1),
    ('Andy Murray', 1),
    -- Men: Next Gen / Top Players
    ('Carlos Alcaraz', 1),
    ('Jannik Sinner', 1),
    ('Daniil Medvedev', 1),
    ('Alexander Zverev', 1),
    ('Andrey Rublev', 1),
    ('Holger Rune', 1),
    ('Taylor Fritz', 1),
    ('Casper Ruud', 1),
    ('Alex de Minaur', 1),
    ('Grigor Dimitrov', 1),
    ('Hubert Hurkacz', 1),
    ('Stefanos Tsitsipas', 1),
    ('Ben Shelton', 1),
    ('Tommy Paul', 1),
    ('Lorenzo Musetti', 1),
    ('Frances Tiafoe', 1),
    ('Karen Khachanov', 1),
    ('Cameron Norrie', 1),
    ('Alexander Bublik', 1),
    ('Nicolas Jarry', 1),
    ('Felix Auger-Aliassime', 1),
    ('Sebastian Baez', 1),
    ('Alejandro Davidovich Fokina', 1),
    ('Francisco Cerundolo', 1),
    ('Jan-Lennard Struff', 1),
    ('Tallon Griekspoor', 1),
    -- Women: Top Players
    ('Iga Swiatek', 1),
    ('Aryna Sabalenka', 1),
    ('Coco Gauff', 1),
    ('Elena Rybakina', 1),
    ('Jessica Pegula', 1),
    ('Ons Jabeur', 1),
    ('Marketa Vondrousova', 1),
    ('Maria Sakkari', 1),
    ('Karolina Muchova', 1),
    ('Barbora Krejcikova', 1),
    ('Jelena Ostapenko', 1),
    ('Daria Kasatkina', 1),
    ('Liudmila Samsonova', 1),
    ('Veronika Kudermetova', 1),
    ('Beatriz Haddad Maia', 1),
    ('Petra Kvitova', 1),
    ('Caroline Garcia', 1),
    ('Madison Keys', 1),
    ('Elina Svitolina', 1),
    ('Ekaterina Alexandrova', 1),
    ('Victoria Azarenka', 1),
    ('Emma Navarro', 1),
    ('Qinwen Zheng', 1),
    ('Jasmine Paolini', 1),
    ('Marta Kostyuk', 1);
