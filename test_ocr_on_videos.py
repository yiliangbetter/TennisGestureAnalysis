#!/usr/bin/env python3
"""
Test script to process videos in raw_videos directory and output
detected player names along with player information from database.

Uses both filename extraction and OCR fallback.
"""

import os
import sys
import re
from pathlib import Path
from database.ocr import VideoTextExtractor
from database_manager import TennisDatabase
from video_processor import VideoProcessor


def process_videos(raw_videos_dir: str = "raw_videos", db_path: str = "tennis_gesture.db"):
    """Process all videos and display detected player names"""

    # Initialize database, OCR extractor, and video processor
    db = TennisDatabase(db_path, include_seed_data=True)
    extractor = VideoTextExtractor(db_path, use_easyocr=True)
    processor = VideoProcessor(db_path, use_ocr=True)

    # Get video files
    raw_dir = Path(raw_videos_dir)
    if not raw_dir.exists():
        print(f"Error: Directory '{raw_videos_dir}' does not exist.")
        return

    video_files = list(raw_dir.glob("*.mp4"))
    video_files.extend(raw_dir.glob("*.mov"))
    video_files.extend(raw_dir.glob("*.avi"))

    if not video_files:
        print(f"No video files found in '{raw_videos_dir}'")
        return

    print("="*80)
    print(f"Processing {len(video_files)} video(s) in '{raw_videos_dir}'")
    print("="*80)
    print()

    results = []

    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*80}")
        print(f"Video {i}/{len(video_files)}: {video_path.name}")
        print(f"{'='*80}")

        filename = video_path.name

        # Step 1: Try filename extraction first
        print("\n[1] Trying filename-based extraction...")
        player_name = None

        # Try mapping first
        PLAYER_NAME_MAPPING = {
            "djokovic": "Novak Djokovic",
            "alcaraz": "Carlos Alcaraz",
            "sinner": "Jannik Sinner",
            "zverev": "Alexander Zverev",
            "rublev": "Andrey Rublev",
            "shelton": "Ben Shelton",
            "nadal": "Rafael Nadal",
            "federer": "Roger Federer",
        }
        filename_lower = filename.lower()
        for key, name in PLAYER_NAME_MAPPING.items():
            if key in filename_lower:
                player_name = name
                print(f"    Found in filename: {player_name}")
                break

        # Step 2: Try OCR if filename didn't yield a known player
        if not player_name:
            print("    Not found in filename, trying OCR...")
            player_name = extractor.extract_player_name_from_video(str(video_path))
            if player_name:
                print(f"    OCR detected: {player_name}")
        else:
            # Still run OCR to populate database, but don't override filename result
            print("    Running OCR to populate database (filename result will be used)...")
            extractor.extract_player_name_from_video(str(video_path))

        # Fallback to capitalized name in filename
        if not player_name:
            matches = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+', filename)
            if matches:
                player_name = matches[0]
                print(f"    Fallback to capitalized name: {player_name}")

        # Final fallback to filename stem
        if not player_name:
            player_name = Path(filename).stem.replace('-', ' ').replace('_', ' ').title()
            print(f"    Using filename stem: {player_name}")

        # Get player info from database
        print(f"\n[2] Player Information from Database...")
        player = db.get_player_by_name(player_name) if player_name else None

        if player:
            print(f"    Player ID: {player['id']}")
            print(f"    Name: {player['name']}")
            print(f"    Country: {player['country'] or 'N/A'}")
            print(f"    Professional: {'Yes' if player['is_professional'] else 'No'}")
            if player.get('metadata'):
                import json
                try:
                    metadata = json.loads(player['metadata'])
                    print(f"    Metadata: {metadata}")
                except:
                    print(f"    Metadata: {player['metadata']}")
        else:
            print(f"    Player '{player_name}' not in database (adding now...)")
            # Add player to database
            player_id = db.add_player(player_name, is_professional=False)
            player = db.get_player(player_id)
            print(f"    Added with ID: {player_id}")

        # Get OCR results summary
        print(f"\n[3] OCR Detection Summary...")
        video_id = None
        with db.connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM videos WHERE file_path = ?",
                (str(video_path),)
            )
            row = cursor.fetchone()
            if row:
                video_id = row['id']

        if video_id:
            ocr_results = extractor.get_detected_text(video_id, player_names_only=False)
            player_name_results = extractor.get_detected_text(video_id, player_names_only=True)

            print(f"    Total text detected: {len(ocr_results)} items")
            print(f"    Player name matches: {len(player_name_results)} items")

            # Show player-related detections
            if ocr_results:
                # Find any text that might be player-related
                potential_names = []
                for r in ocr_results:
                    text = r['detected_text']
                    # Check if text contains any known player last names
                    last_names = ['djokovic', 'alcaraz', 'sinner', 'zverev', 'rublev',
                                  'shelton', 'nadal', 'federer', 'murray', 'swiatek', 'gauff']
                    for last in last_names:
                        if last in text.lower():
                            potential_names.append(r)
                            break

                if potential_names:
                    print(f"\n    Player-related text detected:")
                    for r in potential_names[:5]:
                        print(f"      - Frame {r['frame_number']}: '{r['detected_text']}' (conf: {r['confidence']:.2f})")

        # Store result
        results.append({
            'video': video_path.name,
            'detected_player': player_name,
            'player_info': player
        })

    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for r in results:
        print(f"\nVideo: {r['video']}")
        print(f"  Detected Player: {r['detected_player'] or 'N/A'}")
        if r['player_info']:
            print(f"  Country: {r['player_info']['country'] or 'N/A'}")
            print(f"  Professional: {'Yes' if r['player_info']['is_professional'] else 'No'}")

    print(f"\n{'='*80}")
    print("Database Statistics:")
    print(f"{'='*80}")
    stats = db.get_statistics()
    print(f"  Total Players: {stats['total_players']} ({stats['professional_players']} professionals)")
    print(f"  Total Videos: {stats['total_videos']} ({stats['processed_videos']} processed)")
    print(f"  Total Poses: {stats['total_poses']} ({stats['sample_poses']} sample poses)")

    db.close()


if __name__ == "__main__":
    process_videos()
