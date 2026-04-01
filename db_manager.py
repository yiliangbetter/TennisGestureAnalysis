#!/usr/bin/env python3
"""
Database Visualization Tool for Tennis Gesture Analysis

Usage:
    python visualize_database.py [table_name]

Examples:
    python visualize_database.py              # Show summary of all tables
    python visualize_database.py players      # Show all players
    python visualize_database.py videos       # Show all videos
    python visualize_database.py candidates   # Show candidate players (OCR discoveries)
"""

import sqlite3
import sys
from tabulate import tabulate


def get_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect('tennis_gesture.db')
    conn.row_factory = sqlite3.Row
    return conn


def show_summary():
    """Show summary of all tables"""
    conn = get_connection()

    print("=" * 70)
    print("TENNIS GESTURE DATABASE - SUMMARY")
    print("=" * 70)

    # Players
    cursor = conn.execute("SELECT COUNT(*) as count FROM players")
    total_players = cursor.fetchone()['count']

    cursor = conn.execute("SELECT COUNT(*) as count FROM players WHERE is_professional = 1")
    pro_players = cursor.fetchone()['count']

    cursor = conn.execute("SELECT COUNT(*) as count FROM players WHERE metadata LIKE '%candidate%'")
    candidates = cursor.fetchone()['count']

    print(f"\n🎾 PLAYERS:")
    print(f"   Total: {total_players}")
    print(f"   Professional: {pro_players}")
    print(f"   Candidates (OCR discoveries): {candidates}")

    # Videos
    cursor = conn.execute("SELECT COUNT(*) as count FROM videos")
    total_videos = cursor.fetchone()['count']

    cursor = conn.execute("SELECT COUNT(*) as count FROM videos WHERE processed = 1")
    processed_videos = cursor.fetchone()['count']

    cursor = conn.execute("SELECT SUM(duration_sec) as total FROM videos")
    total_duration = cursor.fetchone()['total'] or 0

    print(f"\n🎬 VIDEOS:")
    print(f"   Total: {total_videos}")
    print(f"   Processed: {processed_videos}")
    print(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")

    # OCR Results
    cursor = conn.execute("SELECT COUNT(*) as count FROM video_text_ocr")
    total_ocr = cursor.fetchone()['count']

    cursor = conn.execute("SELECT COUNT(*) as count FROM video_text_ocr WHERE is_player_name = 1")
    player_ocr = cursor.fetchone()['count']

    print(f"\n🔤 OCR RESULTS:")
    print(f"   Total detections: {total_ocr}")
    print(f"   Matched players: {player_ocr}")

    # Recent candidates
    cursor = conn.execute("""
        SELECT name, metadata, created_at
        FROM players
        WHERE metadata LIKE '%candidate%'
        ORDER BY created_at DESC
        LIMIT 5
    """)
    candidates = cursor.fetchall()

    if candidates:
        print(f"\n🔍 RECENT CANDIDATES:")
        for c in candidates:
            meta = c['metadata'][:50] + '...' if len(c['metadata']) > 50 else c['metadata']
            print(f"   - {c['name']} ({c['created_at']})")

    print("\n" + "=" * 70)
    conn.close()


def show_players(professional_only=False, candidates_only=False):
    """Show players table"""
    conn = get_connection()

    if candidates_only:
        cursor = conn.execute("""
            SELECT id, name, country, is_professional, metadata, created_at
            FROM players
            WHERE metadata LIKE '%candidate%'
            ORDER BY created_at DESC
        """)
        title = "CANDIDATE PLAYERS (OCR Discoveries)"
    elif professional_only:
        cursor = conn.execute("""
            SELECT id, name, country, is_professional, created_at
            FROM players
            WHERE is_professional = 1
            ORDER BY name
        """)
        title = "PROFESSIONAL PLAYERS"
    else:
        cursor = conn.execute("""
            SELECT id, name, country, is_professional, created_at
            FROM players
            ORDER BY is_professional DESC, name
        """)
        title = "ALL PLAYERS"

    rows = cursor.fetchall()
    conn.close()

    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(f"Total: {len(rows)} players\n")

    if not rows:
        print("No players found.")
        return

    # Format for tabulate
    headers = ["ID", "Name", "Country", "Pro", "Created"]
    table_data = []
    for row in rows:
        if candidates_only:
            table_data.append([
                row['id'],
                row['name'][:30],
                row['country'] or 'UNK',
                'Yes' if row['is_professional'] else 'No',
                row['created_at'][:10]
            ])
        else:
            table_data.append([
                row['id'],
                row['name'][:30],
                row['country'] or 'UNK',
                'Yes' if row['is_professional'] else 'No',
                row['created_at'][:10]
            ])

    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()


def show_videos():
    """Show videos table"""
    conn = get_connection()

    cursor = conn.execute("""
        SELECT v.id, v.filename, v.duration_sec, v.total_frames, v.processed,
               COUNT(DISTINCT o.id) as ocr_count,
               SUM(CASE WHEN o.is_player_name = 1 THEN 1 ELSE 0 END) as player_matches
        FROM videos v
        LEFT JOIN video_text_ocr o ON v.id = o.video_id
        GROUP BY v.id
        ORDER BY v.created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    print(f"\n{'=' * 70}")
    print("VIDEOS")
    print(f"{'=' * 70}")
    print(f"Total: {len(rows)} videos\n")

    if not rows:
        print("No videos found.")
        return

    headers = ["ID", "Filename", "Duration", "Frames", "OCR", "Players", "Processed"]
    table_data = []
    for row in rows:
        table_data.append([
            row['id'],
            row['filename'][:25],
            f"{row['duration_sec']:.1f}s" if row['duration_sec'] else "N/A",
            row['total_frames'] or "N/A",
            row['ocr_count'] or 0,
            row['player_matches'] or 0,
            'Yes' if row['processed'] else 'No'
        ])

    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()


def show_ocr():
    """Show OCR results"""
    conn = get_connection()

    cursor = conn.execute("""
        SELECT o.id, v.filename, o.frame_number, o.detected_text,
               o.confidence, o.is_player_name
        FROM video_text_ocr o
        JOIN videos v ON o.video_id = v.id
        ORDER BY o.is_player_name DESC, o.confidence DESC
        LIMIT 50
    """)

    rows = cursor.fetchall()
    conn.close()

    print(f"\n{'=' * 70}")
    print("OCR RESULTS (Top 50)")
    print(f"{'=' * 70}")
    print(f"Total shown: {len(rows)}\n")

    if not rows:
        print("No OCR results found.")
        return

    headers = ["ID", "Video", "Frame", "Text", "Conf", "Player?"]
    table_data = []
    for row in rows:
        table_data.append([
            row['id'],
            row['filename'][:15],
            row['frame_number'],
            row['detected_text'][:20],
            f"{row['confidence']:.2f}",
            'Yes' if row['is_player_name'] else 'No'
        ])

    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()


def delete_player(player_name):
    """Delete a player from the database by name"""
    conn = get_connection()

    # First check if player exists
    cursor = conn.execute(
        "SELECT id, name FROM players WHERE LOWER(name) = LOWER(?)",
        (player_name,)
    )
    player = cursor.fetchone()

    if not player:
        print(f"❌ Player '{player_name}' not found in database")
        conn.close()
        return False

    print(f"\nFound player to delete:")
    print(f"  ID: {player['id']}")
    print(f"  Name: {player['name']}")

    # Confirm deletion
    confirm = input(f"\n⚠️  Are you sure you want to delete '{player['name']}'? [y/N]: ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("❌ Deletion cancelled")
        conn.close()
        return False

    try:
        # Delete the player
        conn.execute("DELETE FROM players WHERE id = ?", (player['id'],))
        conn.commit()
        print(f"✅ Successfully deleted '{player['name']}' (ID: {player['id']})")
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error deleting player: {e}")
        conn.rollback()
        conn.close()
        return False


def delete_candidate_players():
    """Delete all candidate players (OCR discoveries)"""
    conn = get_connection()

    cursor = conn.execute(
        "SELECT id, name FROM players WHERE metadata LIKE '%candidate%'"
    )
    candidates = cursor.fetchall()

    if not candidates:
        print("❌ No candidate players to delete")
        conn.close()
        return False

    print(f"\nFound {len(candidates)} candidate player(s) to delete:")
    for c in candidates:
        print(f"  - {c['name']} (ID: {c['id']})")

    confirm = input(f"\n⚠️  Delete all {len(candidates)} candidates? [y/N]: ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("❌ Deletion cancelled")
        conn.close()
        return False

    try:
        conn.execute("DELETE FROM players WHERE metadata LIKE '%candidate%'")
        conn.commit()
        print(f"✅ Successfully deleted {len(candidates)} candidate player(s)")
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error deleting candidates: {e}")
        conn.rollback()
        conn.close()
        return False


def show_help():
    """Show help message"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           TENNIS GESTURE DATABASE MANAGER                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  USAGE:                                                              ║
║    python db_manager.py <command> [options]                         ║
║                                                                      ║
║  COMMANDS:                                                           ║
║                                                                      ║
║    Viewing Data:                                                     ║
║    ─────────────                                                     ║
║    summary          Show database summary                           ║
║    players          List all players                                ║
║    pros             List professional players only                  ║
║    candidates       List candidate players (OCR discoveries)       ║
║    videos           List all videos                                 ║
║    ocr              List OCR results                                ║
║                                                                      ║
║    Managing Data:                                                    ║
║    ─────────────                                                     ║
║    delete <name>    Delete a specific player by name                ║
║    clean-candidates Delete all candidate players                    ║
║                                                                      ║
║  EXAMPLES:                                                           ║
║    python db_manager.py summary                                     ║
║    python db_manager.py candidates                                  ║
║    python db_manager.py delete "Mercedes Benz"                      ║
║    python db_manager.py clean-candidates                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        show_help()
        return

    command = sys.argv[1].lower()

    # View commands
    if command == 'summary' or command == 's':
        show_summary()
    elif command == 'players':
        show_players()
    elif command == 'pros' or command == 'professional':
        show_players(professional_only=True)
    elif command == 'candidates':
        show_players(candidates_only=True)
    elif command == 'videos':
        show_videos()
    elif command == 'ocr':
        show_ocr()

    # Management commands
    elif command == 'delete':
        if len(sys.argv) < 3:
            print("❌ Error: Please specify a player name to delete")
            print("   Example: python db_manager.py delete 'Mercedes Benz'")
            return
        player_name = ' '.join(sys.argv[2:])
        delete_player(player_name)

    elif command == 'clean-candidates':
        delete_candidate_players()

    else:
        print(f"❌ Unknown command: {command}")
        print("\nRun 'python db_manager.py help' for usage information")


if __name__ == '__main__':
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tabulate'])
        from tabulate import tabulate

    main()
