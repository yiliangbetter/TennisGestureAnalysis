#!/usr/bin/env python3
"""
Video Processor for Tennis Gesture Analysis

Processes videos from raw_videos directory, extracts pose data,
and populates the database for comparison queries.
"""

import cv2
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

from database_manager import TennisDatabase, PoseData
from enhanced_gesture_analyzer import EnhancedTennisGestureAnalyzer


class VideoProcessor:
    """
    Processes tennis videos to extract and store pose data.

    Workflow:
    1. Scan raw_videos directory for video files
    2. Extract player name from filename
    3. Process each frame to extract landmarks
    4. Detect stroke type from pose characteristics
    5. Store poses in database
    6. Create sample poses using clustering
    """

    # Player name mapping from filenames
    PLAYER_NAME_MAPPING = {
        "djokovic": "Novak Djokovic",
        "alcaraz": "Carlos Alcaraz",
        "sinner": "Jannik Sinner",
        "zverev": "Alexander Zverev",
        "rublev": "Andrey Rublev",
        "shelton": "Ben Shelton",
        "nadal": "Rafael Nadal",
        "federer": "Roger Federer",
        "serena": "Serena Williams",
        "swiatek": "Iga Swiatek",
        "gauff": "Coco Gauff",
    }

    # Stroke type keywords in filenames
    STROKE_KEYWORDS = {
        "forehand": ["forehand", "fh", "正手"],
        "backhand": ["backhand", "bh", "反手"],
        "serve": ["serve", "serv", "发球"],
        "volley": ["volley", "截击"],
    }

    def __init__(self, db_path: str = "tennis_gesture.db",
                 use_opencv_pose: bool = False):
        """
        Initialize video processor.

        Args:
            db_path: Path to SQLite database
            use_opencv_pose: Use OpenCV-based pose estimation (vs MediaPipe)
        """
        self.db = TennisDatabase(db_path)
        self.analyzer = EnhancedTennisGestureAnalyzer(use_opencv_pose=use_opencv_pose)

    def process_raw_videos(self, raw_videos_dir: str = "raw_videos",
                          skip_processed: bool = True) -> Dict[str, int]:
        """
        Process all videos in the raw_videos directory.

        Args:
            raw_videos_dir: Directory containing raw videos
            skip_processed: Skip videos that have already been processed

        Returns:
            Dict with processing statistics
        """
        raw_dir = Path(raw_videos_dir)
        if not raw_dir.exists():
            print(f"Error: Directory '{raw_videos_dir}' does not exist.")
            return {"processed": 0, "skipped": 0, "errors": 0}

        video_files = list(raw_dir.glob("*.mp4"))
        video_files.extend(raw_dir.glob("*.mov"))
        video_files.extend(raw_dir.glob("*.avi"))

        stats = {"processed": 0, "skipped": 0, "errors": 0}

        print(f"Found {len(video_files)} video files in '{raw_videos_dir}'\n")

        for video_path in video_files:
            print(f"{'='*60}")
            print(f"Processing: {video_path.name}")
            print(f"{'='*60}")

            try:
                # Check if already processed
                if skip_processed:
                    existing = self._check_if_processed(str(video_path))
                    if existing:
                        print(f"  Skipping (already processed)")
                        stats["skipped"] += 1
                        continue

                # Process the video
                result = self._process_single_video(str(video_path))
                if result:
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                stats["errors"] += 1

        print(f"\n{'='*60}")
        print(f"Processing Complete!")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")
        print(f"{'='*60}")

        return stats

    def _check_if_processed(self, video_path: str) -> bool:
        """Check if video has already been processed"""
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT processed FROM videos WHERE file_path = ?",
                (video_path,)
            )
            row = cursor.fetchone()
            return row and row['processed']

    def _process_single_video(self, video_path: str) -> Optional[int]:
        """
        Process a single video file.

        Args:
            video_path: Absolute path to video file

        Returns:
            Video ID if successful, None otherwise
        """
        # Extract metadata from filename
        filename = os.path.basename(video_path)
        player_name = self._extract_player_name(filename)
        stroke_type = self._extract_stroke_type(filename)

        print(f"  Player: {player_name}")
        print(f"  Stroke: {stroke_type}")

        # Add/get player ID
        player_id = self.db.add_player(
            name=player_name,
            is_professional=True
        )

        # Add video and get ID
        video_id = self._add_video_to_db(video_path)
        if not video_id:
            print(f"  ERROR: Could not add video to database")
            return None

        # Process frames
        frame_count = self._process_video_frames(
            video_path, video_id, player_id, stroke_type
        )

        print(f"  Processed {frame_count} frames")

        # Create sample poses using clustering
        self._create_sample_poses(video_id, player_id, stroke_type)

        # Mark video as processed
        self.db.mark_video_processed(video_id)

        return video_id

    def _extract_player_name(self, filename: str) -> str:
        """Extract player name from video filename"""
        filename_lower = filename.lower()

        # Try mapping first
        for key, name in self.PLAYER_NAME_MAPPING.items():
            if key in filename_lower:
                return name

        # Try to find capitalized words
        matches = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+', filename)
        if matches:
            return matches[0]

        # Fall back to filename without extension
        return Path(filename).stem.replace('-', ' ').replace('_', ' ').title()

    def _extract_stroke_type(self, filename: str) -> str:
        """Extract stroke type from video filename"""
        filename_lower = filename.lower()

        for stroke_type, keywords in self.STROKE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return stroke_type

        # Default to forehand
        return "forehand"

    def _add_video_to_db(self, video_path: str) -> Optional[int]:
        """Add video metadata to database"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        metadata = {
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames
        }

        cap.release()

        return self.db.add_video(
            filename=os.path.basename(video_path),
            file_path=video_path,
            metadata=metadata
        )

    def _process_video_frames(self, video_path: str, video_id: int,
                              player_id: int, stroke_type: str) -> int:
        """
        Process each frame and extract poses.

        Args:
            video_path: Path to video file
            video_id: Database video ID
            player_id: Database player ID
            stroke_type: Detected stroke type

        Returns:
            Number of frames with valid poses
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        valid_pose_count = 0

        # Reset analyzer state for new video
        self.analyzer.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract landmarks using the analyzer
            landmarks = self.analyzer.extract_landmarks_from_frame(frame)

            if landmarks is not None and len(landmarks) >= 33:
                # Calculate joint angles
                angles_list = self.analyzer.calculate_joint_angles(landmarks)

                # Map to named angles
                angle_dict = self._map_angles(angles_list)

                # Create pose data
                pose_data = PoseData(
                    id=None,
                    video_id=video_id,
                    frame_number=frame_num,
                    player_id=player_id,
                    landmarks=landmarks,
                    joint_angles=angle_dict,
                    bbox=None,
                    stroke_type=stroke_type,
                    confidence=1.0,
                    timestamp_ms=frame_num / fps * 1000 if fps > 0 else None
                )

                # Add to database
                self.db.add_pose(pose_data)
                valid_pose_count += 1

            frame_num += 1

            # Progress indicator
            if frame_num % 50 == 0:
                print(f"    Frame {frame_num}...")

        cap.release()

        return valid_pose_count

    def _map_angles(self, angles_list: List[float]) -> Dict[str, float]:
        """Map angle list to named dictionary"""
        angle_names = [
            'right_elbow_flexion',
            'right_shoulder_abduction',
            'left_elbow_flexion',
            'left_shoulder_abduction',
            'right_knee_flexion',
            'right_hip_angle',
            'left_knee_flexion',
            'left_hip_angle',
            'torso_rotation_right',
            'torso_rotation_left',
            'body_lean_right',
            'body_lean_left',
        ]

        return {
            name: angles_list[i] if i < len(angles_list) else 0.0
            for i, name in enumerate(angle_names)
        }

    def _create_sample_poses(self, video_id: int, player_id: int,
                            stroke_type: str, n_clusters: int = 5):
        """
        Create sample poses from processed video using clustering.

        Selects representative poses that best represent the stroke.

        Args:
            video_id: Database video ID
            player_id: Database player ID
            stroke_type: Type of stroke
            n_clusters: Number of sample poses to create
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("  WARNING: scikit-learn not available, skipping sample pose creation")
            return

        # Get all poses for this video
        poses = self.db.get_poses_by_video(video_id)

        if len(poses) < n_clusters:
            print(f"  Not enough poses ({len(poses)}) for clustering")
            return

        # Flatten landmarks for clustering
        all_landmarks = []
        valid_indices = []

        for i, pose in enumerate(poses):
            if pose.landmarks is not None and pose.landmarks.size > 0:
                flattened = pose.landmarks.flatten()
                all_landmarks.append(flattened)
                valid_indices.append(i)

        if len(all_landmarks) < n_clusters:
            return

        all_landmarks = np.array(all_landmarks)

        # Cluster poses
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(all_landmarks)

        # Find closest pose to each centroid
        for i, centroid in enumerate(kmeans.cluster_centers_):
            distances = np.linalg.norm(all_landmarks - centroid, axis=1)
            closest_idx = np.argmin(distances)
            closest_pose = poses[valid_indices[closest_idx]]

            # Mark as sample pose
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE extracted_poses SET is_sample_pose = 1 WHERE id = ?",
                    (closest_pose.id,)
                )

                # Calculate frame range for sequence
                frame_range = max(1, len(poses) // 10)
                start_frame = max(0, closest_pose.frame_number - frame_range)
                end_frame = closest_pose.frame_number + frame_range

                # Create gesture sequence
                self.db.add_gesture_sequence(
                    pose_id=closest_pose.id,
                    video_id=video_id,
                    player_id=player_id,
                    sequence_type=stroke_type,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    key_frame=closest_pose.frame_number
                )

            print(f"  Created sample pose {i+1}/{n_clusters} (frame {closest_pose.frame_number})")

    def compare_input_video(self, input_video_path: str,
                           stroke_type: str = None) -> Dict:
        """
        Compare input video against database and return results.

        Args:
            input_video_path: Path to input video
            stroke_type: Optional stroke type filter

        Returns:
            Dict with comparison results
        """
        print(f"\nComparing input video: {input_video_path}")
        print(f"{'='*60}")

        # Process input video frames
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            return {"error": "Could not open video"}

        results = {
            'frame_matches': [],
            'player_scores': {},
            'best_match': None,
            'recommendations': []
        }

        frame_num = 0
        all_similarities = []

        print("  Analyzing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract landmarks from input frame
            landmarks = self.analyzer.extract_landmarks_from_frame(frame)

            if landmarks is not None and len(landmarks) >= 33:
                # Find similar poses in database
                similar = self.db.find_similar_poses(
                    landmarks, stroke_type=stroke_type, top_k=3
                )

                if similar:
                    for pose, sim in similar:
                        player_id = pose.player_id
                        if player_id not in results['player_scores']:
                            results['player_scores'][player_id] = []
                        results['player_scores'][player_id].append(sim)
                        all_similarities.append(sim)

                    # Store best match for this frame
                    best_pose, best_sim = similar[0]
                    results['frame_matches'].append({
                        'frame': frame_num,
                        'matched_player_id': best_pose.player_id,
                        'similarity': best_sim
                    })

            frame_num += 1

            if frame_num % 30 == 0:
                print(f"    Frame {frame_num}...")

        cap.release()

        # Calculate overall best match
        if results['player_scores']:
            player_avgs = {
                pid: np.mean(scores)
                for pid, scores in results['player_scores'].items()
            }

            best_player_id = max(player_avgs.keys(), key=lambda p: player_avgs[p])
            best_player = self.db.get_player(best_player_id)

            results['best_match'] = {
                'player_id': best_player_id,
                'player_name': best_player['name'] if best_player else 'Unknown',
                'avg_similarity': player_avgs[best_player_id],
                'max_similarity': max(player_avgs.values())
            }

            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(
                player_avgs, all_similarities
            )

        # Save results to database
        if results['best_match']:
            self.db.save_comparison_result(
                input_video_path=input_video_path,
                matched_player_id=results['best_match']['player_id'],
                similarity_score=results['best_match']['avg_similarity'],
                recommendations=results['recommendations']
            )

        # Print results
        self._print_results(results)

        return results

    def _generate_recommendations(self, player_scores: Dict,
                                  all_similarities: List[float]) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []

        avg_sim = np.mean(all_similarities) if all_similarities else 0

        if avg_sim > 0.8:
            recommendations.append("Excellent! Your form closely matches professional technique.")
        elif avg_sim > 0.6:
            recommendations.append("Good form! Focus on refining your technique for better consistency.")
        elif avg_sim > 0.4:
            recommendations.append("Consider working on your basic stance and swing path.")
        else:
            recommendations.append("Focus on fundamental technique - consider coaching sessions.")

        # Add specific recommendations based on lowest scoring frames
        if all_similarities:
            low_frames = [s for s in all_similarities if s < 0.5]
            if len(low_frames) > len(all_similarities) * 0.3:
                recommendations.append(
                    "Over 30% of frames show significant deviation - "
                    "slow down and focus on form consistency."
                )

        return recommendations

    def _print_results(self, results: Dict):
        """Print comparison results"""
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")

        if results['best_match']:
            bm = results['best_match']
            print(f"\nBest Match: {bm['player_name']}")
            print(f"Average Similarity: {bm['avg_similarity']:.1%}")
            print(f"Peak Similarity: {bm['max_similarity']:.1%}")

            print(f"\nPlayer Scores:")
            for pid, scores in sorted(results['player_scores'].items(),
                                     key=lambda x: np.mean(x[1]), reverse=True):
                player = self.db.get_player(pid)
                name = player['name'] if player else f"Player {pid}"
                print(f"  {name}: {np.mean(scores):.1%}")

            print(f"\nRecommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print("No matches found in database.")

        print(f"\n{'='*60}")


def main():
    """Main entry point for video processing"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Process tennis videos and compare against database'
    )
    parser.add_argument(
        '--process-raw',
        action='store_true',
        help='Process all videos in raw_videos directory'
    )
    parser.add_argument(
        '--compare',
        type=str,
        help='Compare an input video against the database'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='tennis_gesture.db',
        help='Database file path'
    )
    parser.add_argument(
        '--stroke',
        type=str,
        choices=['forehand', 'backhand', 'serve', 'volley'],
        help='Filter by stroke type'
    )

    args = parser.parse_args()

    processor = VideoProcessor(db_path=args.db)

    if args.process_raw:
        processor.process_raw_videos("raw_videos")
    elif args.compare:
        processor.compare_input_video(args.compare, stroke_type=args.stroke)
    else:
        # Default: show database statistics
        stats = processor.db.get_statistics()
        print("\nDatabase Statistics:")
        print(f"  Players: {stats['total_players']} ({stats['professional_players']} professionals)")
        print(f"  Videos: {stats['total_videos']} ({stats['processed_videos']} processed)")
        print(f"  Poses: {stats['total_poses']} ({stats['sample_poses']} sample poses)")
        print(f"  Landmarks: {stats['total_landmarks']}")
        print(f"  Angles: {stats['total_angles']}")

        if stats['poses_by_stroke']:
            print(f"\nPoses by Stroke Type:")
            for stroke, count in stats['poses_by_stroke'].items():
                print(f"  {stroke}: {count}")


if __name__ == "__main__":
    main()
