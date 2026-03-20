# Database Quick Start Guide

This guide shows you how to use the SQLite database for tennis gesture analysis.

## Quick Start

### Step 1: Initialize the Database

The database is automatically created when you first run the video processor. No manual setup needed.

```bash
# The database file (tennis_gesture.db) will be created automatically
python video_processor.py
```

### Step 2: Process Raw Videos

Process all videos in the `raw_videos/` directory to extract pose data:

```bash
python video_processor.py --process-raw
```

This will:
1. Scan `raw_videos/` for video files
2. Extract player names from filenames
3. Detect stroke types (forehand/backhand/serve)
4. Process each frame to extract 33 MediaPipe landmarks
5. Calculate joint angles
6. Store everything in the database
7. Create sample poses using K-Means clustering

**Example output:**
```
Found 3 video files in 'raw_videos'

============================================================
Processing: tennis-djokovic-forehand.mp4
============================================================
  Player: Novak Djokovic
  Stroke: forehand
    Frame 50...
    Frame 100...
  Processed 187 frames
  Created sample pose 1/5 (frame 42)
  Created sample pose 2/5 (frame 67)
  ...
```

### Step 3: Compare Input Video

Compare your video against the database of professional players:

```bash
python video_processor.py --compare SampleInputVideos.mp4
```

**Optional: Filter by stroke type:**
```bash
python video_processor.py --compare my_forehand.mp4 --stroke forehand
```

### Step 4: View Database Statistics

```bash
python video_processor.py --db tennis_gesture.db
```

**Example output:**
```
Database Statistics:
  Players: 6 (6 professionals)
  Videos: 3 (3 processed)
  Poses: 542 (15 sample poses)
  Landmarks: 17886
  Angles: 6504

Poses by Stroke Type:
  forehand: 542
```

---

## Using the Database in Your Code

### Initialize Database

```python
from database_manager import TennisDatabase

db = TennisDatabase("tennis_gesture.db")
```

### Add a Player

```python
# Add a professional player
player_id = db.add_player(
    name="Your Name",
    country="USA",
    is_professional=False
)
```

### Add a Video

```python
video_id = db.add_video(
    filename="my_tennis_video.mp4",
    file_path="/path/to/video.mp4",
    metadata={
        'duration': 30.5,
        'fps': 30,
        'width': 1920,
        'height': 1080,
        'total_frames': 915
    }
)
```

### Add a Pose

```python
import numpy as np
from database_manager import PoseData

# Create pose data
pose_data = PoseData(
    id=None,
    video_id=video_id,
    frame_number=42,
    player_id=player_id,
    landmarks=np.array([...]),  # (33, 2) array
    joint_angles={
        'right_elbow_flexion': 95.5,
        'right_shoulder_abduction': 45.2,
        # ... more angles
    },
    stroke_type="forehand",
    confidence=0.95
)

# Add to database
pose_id = db.add_pose(pose_data)
```

### Query Sample Poses

```python
# Get all sample poses
sample_poses = db.get_sample_poses()

# Filter by player
djokovic_poses = db.get_sample_poses(player_name="Novak Djokovic")

# Filter by stroke type
forehand_poses = db.get_sample_poses(stroke_type="forehand")

# Access pose data
for pose in forehand_poses:
    print(f"Frame {pose.frame_number}: {pose.landmarks.shape}")
    print(f"Joint angles: {pose.joint_angles}")
```

### Find Similar Poses

```python
# Find poses similar to input landmarks
input_landmarks = np.array([...])  # Your (33, 2) landmark array

similar = db.find_similar_poses(
    input_landmarks,
    stroke_type="forehand",
    top_k=5
)

for pose, similarity in similar:
    print(f"Match: {similarity:.1%}")
    print(f"Player ID: {pose.player_id}")
```

### Calculate Similarity Scores

```python
# Pose similarity (landmarks)
pose_sim = db.calculate_pose_similarity(landmarks1, landmarks2)
print(f"Pose similarity: {pose_sim:.1%}")

# Angle similarity
angle_sim = db.calculate_angle_similarity(angles1, angles2)
print(f"Angle similarity: {angle_sim:.1%}")
```

### Get Comparison History

```python
results = db.get_comparison_history("SampleInputVideos.mp4")
for result in results:
    print(f"Matched: {result['matched_player_id']}")
    print(f"Similarity: {result['similarity_score']:.1%}")
```

---

## Direct SQL Queries

You can also query the database directly using SQLite:

```bash
# Open database
sqlite3 tennis_gesture.db
```

```sql
-- List all professional players
SELECT name, country FROM players WHERE is_professional = 1;

-- Count poses by player
SELECT p.name, COUNT(ep.id) as pose_count
FROM players p
LEFT JOIN extracted_poses ep ON p.id = ep.player_id
GROUP BY p.name;

-- Get sample poses with player names
SELECT p.name, ep.stroke_type, ep.frame_number
FROM extracted_poses ep
JOIN players p ON ep.player_id = p.id
WHERE ep.is_sample_pose = 1
ORDER BY p.name, ep.stroke_type;

-- Get joint angles for a specific pose
SELECT angle_type, angle_value
FROM joint_angles
WHERE pose_id = 42;

-- Find poses with high elbow flexion
SELECT ep.id, p.name, ja.angle_value
FROM extracted_poses ep
JOIN players p ON ep.player_id = p.id
JOIN joint_angles ja ON ep.id = ja.pose_id
WHERE ja.angle_type = 'right_elbow_flexion'
  AND ja.angle_value > 120;
```

---

## Database File Management

### Backup Database

```bash
# Copy database file
cp tennis_gesture.db tennis_gesture_backup.db

# Or use SQLite dump
sqlite3 tennis_gesture.db .dump > backup.sql
```

### Restore Database

```bash
# From file backup
cp tennis_gesture_backup.db tennis_gesture.db

# From SQL dump
sqlite3 tennis_gesture.db < backup.sql
```

### Reset Database

```bash
# Delete and recreate
rm tennis_gesture.db
python video_processor.py  # Will auto-create
```

---

## Troubleshooting

### Database is locked

```bash
# Another process may be using the database
# Wait for the process to complete or kill it
lsof tennis_gesture.db
```

### No sample poses found

Make sure you've processed the raw videos:
```bash
python video_processor.py --process-raw
```

### Import errors

Ensure dependencies are installed:
```bash
pip install numpy scikit-learn opencv-python
```

---

## Performance Tips

1. **Index usage**: The database includes indexes on commonly queried columns
2. **Batch inserts**: Use `executemany()` for inserting multiple poses
3. **Connection pooling**: Reuse the `TennisDatabase` instance
4. **Limit queries**: Always use `LIMIT` for large result sets
5. **Filter early**: Use `stroke_type` and `player_name` filters in `get_sample_poses()`

---

## File Reference

| File | Purpose |
|------|---------|
| `schema.sql` | SQL schema definition |
| `database_manager.py` | Python database API |
| `video_processor.py` | Video processing utility |
| `DATABASE_DESIGN.md` | Detailed design document |
| `tennis_gesture.db` | SQLite database file (auto-created) |
