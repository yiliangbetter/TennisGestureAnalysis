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
2. **Feature Extraction**: Extracts key gesture features including:
   - Joint positions (shoulders, elbows, wrists, hips, knees, ankles)
   - Joint angles
   - Movement trajectories
   - Velocity and acceleration vectors
3. **Comparison**: Compares extracted features with a database of professional tennis players
4. **Analysis**: Calculates similarity scores and identifies differences
5. **Recommendations**: Provides personalized tips for technique improvement

## Architecture

- `gesture_analyzer.py`: Core gesture analysis and comparison logic
- `main.py`: Main application entry point
- `setup.py`: Installation and dependency management
- `requirements.txt`: Project dependencies

## Database Structure

The gesture database contains:
- Sample gesture sequences from professional players
- Pose landmark data for each frame
- Key joint positions and angles
- Movement trajectories
- Timing information

## Technologies Used

- OpenCV: Video processing and computer vision
- MediaPipe: Human pose estimation
- NumPy: Numerical computations
- scikit-learn: Similarity calculations
- pickle: Data serialization

## Future Enhancements

- Support for more tennis players and strokes
- Real-time analysis capability
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