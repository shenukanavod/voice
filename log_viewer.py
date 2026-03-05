"""
Database Log Viewer
View and analyze logs stored in MongoDB
"""

import argparse
from datetime import datetime, timedelta
from typing import Optional
from db import VoiceDatabase
from app.config import settings


class LogViewer:
    """View and analyze logs from MongoDB."""

    def __init__(self, db: VoiceDatabase):
        self.db = db

    def view_recent_logs(self, limit: int = 50, level: Optional[str] = None):
        """
        View recent logs.

        Args:
            limit: Number of logs to display
            level: Filter by log level (INFO, WARNING, ERROR, etc.)
        """
        print("=" * 100)
        print(f"RECENT LOGS (Limit: {limit})")
        if level:
            print(f"Filter: Level = {level}")
        print("=" * 100)
        print()

        logs = self.db.get_logs(level=level, limit=limit)

        if not logs:
            print("No logs found.")
            return

        for log in logs:
            timestamp = log.get('timestamp', datetime.now())
            level_str = log.get('level', 'INFO')
            module = log.get('module', 'unknown')
            user_id = log.get('user_id', '-')
            message = log.get('message', '')

            # Color code by level
            level_emoji = {
                'DEBUG': '🔍',
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }.get(level_str, '📝')

            print(f"{level_emoji} [{timestamp}] [{level_str:8s}] [{module:20s}] [{user_id:15s}] {message}")

        print()
        print(f"Total logs displayed: {len(logs)}")
        print("=" * 100)

    def view_user_logs(self, user_id: str, limit: int = 50):
        """View logs for a specific user."""
        print("=" * 100)
        print(f"LOGS FOR USER: {user_id} (Limit: {limit})")
        print("=" * 100)
        print()

        logs = self.db.get_logs(user_id=user_id, limit=limit)

        if not logs:
            print(f"No logs found for user: {user_id}")
            return

        for log in logs:
            timestamp = log.get('timestamp', datetime.now())
            level_str = log.get('level', 'INFO')
            module = log.get('module', 'unknown')
            message = log.get('message', '')

            level_emoji = {
                'DEBUG': '🔍',
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }.get(level_str, '📝')

            print(f"{level_emoji} [{timestamp}] [{level_str:8s}] [{module:20s}] {message}")

        print()
        print(f"Total logs displayed: {len(logs)}")
        print("=" * 100)

    def view_errors(self, limit: int = 50):
        """View error and critical logs."""
        print("=" * 100)
        print(f"ERROR & CRITICAL LOGS (Limit: {limit})")
        print("=" * 100)
        print()

        error_logs = self.db.get_logs(level="ERROR", limit=limit)
        critical_logs = self.db.get_logs(level="CRITICAL", limit=limit//2)

        all_errors = error_logs + critical_logs
        all_errors.sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=True)

        if not all_errors:
            print("✅ No errors found!")
            return

        for log in all_errors[:limit]:
            timestamp = log.get('timestamp', datetime.now())
            level_str = log.get('level', 'ERROR')
            module = log.get('module', 'unknown')
            user_id = log.get('user_id', '-')
            message = log.get('message', '')

            level_emoji = '❌' if level_str == 'ERROR' else '🚨'

            print(f"{level_emoji} [{timestamp}] [{level_str:8s}] [{module:20s}] [{user_id:15s}] {message}")

            # Show extra data if available
            extra_data = log.get('extra_data', {})
            if extra_data and isinstance(extra_data, dict):
                for key, value in extra_data.items():
                    if key not in ['filename', 'lineno', 'funcName', 'process', 'thread', 'threadName']:
                        print(f"      {key}: {value}")

        print()
        print(f"Total errors displayed: {len(all_errors[:limit])}")
        print("=" * 100)

    def get_statistics(self):
        """Display log statistics."""
        print("=" * 100)
        print("LOG STATISTICS")
        print("=" * 100)
        print()

        total = self.db.count_logs()
        info = self.db.count_logs("INFO")
        warning = self.db.count_logs("WARNING")
        error = self.db.count_logs("ERROR")
        critical = self.db.count_logs("CRITICAL")
        debug = self.db.count_logs("DEBUG")

        print(f"Total Logs:     {total:,}")
        print(f"  ℹ️  INFO:      {info:,}")
        print(f"  ⚠️  WARNING:   {warning:,}")
        print(f"  ❌ ERROR:     {error:,}")
        print(f"  🚨 CRITICAL:  {critical:,}")
        print(f"  🔍 DEBUG:     {debug:,}")
        print()
        print("=" * 100)

    def cleanup_old_logs(self, days: int = 30):
        """Delete logs older than specified days."""
        print(f"Deleting logs older than {days} days...")
        deleted_count = self.db.delete_old_logs(days)
        print(f"✅ Deleted {deleted_count} old logs")


def main():
    """Main function for log viewer CLI."""
    parser = argparse.ArgumentParser(description="View logs from MongoDB database")
    parser.add_argument('--recent', type=int, help='View N recent logs', metavar='N')
    parser.add_argument('--user', type=str, help='View logs for specific user', metavar='USER_ID')
    parser.add_argument('--errors', action='store_true', help='View error and critical logs')
    parser.add_argument('--stats', action='store_true', help='Show log statistics')
    parser.add_argument('--level', type=str, help='Filter by log level (INFO, WARNING, ERROR, etc.)')
    parser.add_argument('--cleanup', type=int, help='Delete logs older than N days', metavar='DAYS')
    parser.add_argument('--limit', type=int, default=50, help='Limit number of logs to display (default: 50)')

    args = parser.parse_args()

    try:
        # Initialize database
        db = VoiceDatabase(settings.MONGODB_URL)
        viewer = LogViewer(db)

        # Execute requested action
        if args.stats:
            viewer.get_statistics()
        elif args.errors:
            viewer.view_errors(limit=args.limit)
        elif args.user:
            viewer.view_user_logs(args.user, limit=args.limit)
        elif args.recent is not None:
            viewer.view_recent_logs(limit=args.recent, level=args.level)
        elif args.cleanup is not None:
            viewer.cleanup_old_logs(days=args.cleanup)
        else:
            # Default: show recent logs
            viewer.view_recent_logs(limit=args.limit, level=args.level)

        db.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
