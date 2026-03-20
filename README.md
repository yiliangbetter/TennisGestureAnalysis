# Tennis Gesture Analysis

This project analyzes tennis gestures from video input and compares them to a database of professional tennis players' movements. It provides similarity scores and personalized recommendations to improve playing technique.

## Features

- Analyzes tennis movements using computer vision
- Compares user's technique to professional players
- Provides similarity scores and improvement recommendations
- Generates annotated output videos showing analysis
- Supports various tennis strokes (forehand, backhand, serve, etc.)

## Installation

1. Clone this repository
2. Navigate to the project directory
3. Run the setup script:
   ```bash
   python setup.py
   ```

   Or manually install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Note**: This project uses MediaPipe for pose estimation. You'll need to download the pose estimation model separately:
```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

## Usage

```bash
python main.py <input_video_path> [options]
```

### Options:
- `--output, -o`: Path to save the output video with analysis overlays
- `--save_db`: Path to save the gesture database (default: gesture_database.pkl)
- `--load_db`: Path to load an existing gesture database

### Example:
```bash
python main.py my_tennis_video.mp4 --output analysis_output.mp4
```

## How It Works

1. **Video Processing**: The system processes video frames to detect human pose landmarks using MediaPipe
2. **Enhanced Feature Extraction**: Extracts key gesture features including:
   - Joint positions (shoulders, elbows, wrists, hips, knees, ankles)
   - Joint angles
   - Movement trajectories
   - Velocity and acceleration vectors
   - Optical flow for motion patterns
   - Motion history images
   - Histogram of Oriented Gradients (HOG) features
3. **Comparison**: Compares extracted features with a database of professional tennis players
4. **Analysis**: Calculates similarity scores and identifies differences
5. **Recommendations**: Provides personalized tips for technique improvement

## Architecture

- `enhanced_gesture_analyzer.py`: Enhanced gesture analysis and comparison logic with multiple feature extraction methods
- `main.py`: Main application entry point
- `setup.py`: Installation and dependency management
- `add_gesture.py`: Utility to add new gestures to the database
- `demo.py`: Demonstration script
- `test_system.py`: System verification script
- `requirements.txt`: Project dependencies

## Database (SQLite)

### New Database Files

- `schema.sql`: Complete SQLite schema definition
- `database_manager.py`: Python database manager class
- `video_processor.py`: Video processing and database population
- `DATABASE_DESIGN.md`: Detailed database design documentation

### Database Tables

| Table | Description |
|-------|-------------|
| `players` | Player information (name, country, professional status) |
| `videos` | Video metadata (path, duration, fps, dimensions) |
| `extracted_poses` | Pose data per frame (stroke type, confidence, BLOBs) |
| `pose_landmarks` | 33 MediaPipe landmarks per pose (x, y, visibility) |
| `joint_angles` | Pre-calculated joint angles for fast comparison |
| `gesture_sequences` | Complete gesture sequences (forehand, backhand, serve) |
| `comparison_results` | Cached comparison results |

### Processing Workflow

```bash
# 1. Process all videos in raw_videos/ to populate database
python video_processor.py --process-raw

# 2. Compare an input video against the database
python video_processor.py --compare SampleInputVideos.mp4

# 3. View database statistics
python video_processor.py --db tennis_gesture.db

# 4. Filter by stroke type
python video_processor.py --compare my_forehand.mp4 --stroke forehand
```

### Sample Poses

The system automatically creates sample poses from raw_videos using K-Means clustering:
- 5 representative poses per video
- Marked with `is_sample_pose = 1`
- Used for fast similarity comparison

## Technologies Used

- OpenCV: Video processing and computer vision
- MediaPipe: Human pose estimation
- NumPy: Numerical computations
- scikit-learn: Similarity calculations
- pickle: Data serialization

## Future Enhancements

- Integration with real-time pose estimation
- Advanced biomechanical analysis
- Integration with wearable sensors
- Performance tracking over time
- Custom training data addition

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Disclaimer

This project is for educational and training purposes. Always consult with professional coaches for advanced tennis technique training.