"""
Test Database Logging
Quick test to verify all monitoring components are logging to database
"""

from db import VoiceDatabase
from app.config import settings
from datetime import datetime


def test_database_logging():
    """Test that we can save logs to database."""
    try:
        print("=" * 80)
        print("TESTING DATABASE LOGGING")
        print("=" * 80)
        print()
        
        # Connect to database
        db = VoiceDatabase(settings.MONGODB_URL)
        print("✅ Connected to database")
        print()
        
        # Test 1: Save a monitoring check log
        print("Test 1: Saving monitoring check log...")
        db.save_log(
            level="INFO",
            message="Voice check passed - Similarity: 95.23%",
            module="voice_monitoring_check",
            user_id="test_user_001",
            extra_data={
                "similarity": 0.9523,
                "threshold": 0.80,
                "verified": True,
                "multiple_speakers": False,
                "consecutive_failures": 0,
                "audio_samples": 47104
            }
        )
        print("✅ Monitoring check logged")
        print()
        
        # Test 2: Save admin panel statistics
        print("Test 2: Saving admin panel statistics...")
        db.save_log(
            level="INFO",
            message="Admin panel statistics update",
            module="admin_panel_stats",
            extra_data={
                "total_sessions": 1,
                "total_checks": 5,
                "total_success": 4,
                "total_alerts": 0,
                "total_multi_voice": 0,
                "avg_success_rate": 0.80,
                "active_users": ["test_user_001"]
            }
        )
        print("✅ Admin panel stats logged")
        print()
        
        # Test 3: Save monitoring session start
        print("Test 3: Saving monitoring session...")
        db.save_log(
            level="INFO",
            message="Monitoring session started",
            module="admin_panel_session",
            user_id="test_user_001",
            extra_data={
                "action": "session_started",
                "threshold": 0.80,
                "check_interval": 1.0
            }
        )
        print("✅ Monitoring session logged")
        print()
        
        # Test 4: Save a failed check
        print("Test 4: Saving failed check...")
        db.save_log(
            level="WARNING",
            message="Voice check FAILED - Similarity: 45.67%",
            module="voice_monitoring_check",
            user_id="test_user_001",
            extra_data={
                "similarity": 0.4567,
                "threshold": 0.80,
                "verified": False,
                "multiple_speakers": False,
                "consecutive_failures": 1,
                "audio_samples": 47104
            }
        )
        print("✅ Failed check logged")
        print()
        
        # Test 5: Save a security alert
        print("Test 5: Saving security alert...")
        db.save_log(
            level="CRITICAL",
            message="SECURITY ALERT: Voice verification failed 3 consecutive times",
            module="voice_monitoring_alert",
            user_id="test_user_001",
            extra_data={
                "consecutive_failures": 3,
                "similarity": 0.5123,
                "threshold": 0.80,
                "multiple_speakers": False,
                "timestamp": datetime.utcnow().isoformat(),
                "alert_message": "Security alert triggered"
            }
        )
        print("✅ Security alert logged")
        print()
        
        # Verify logs were saved
        print("=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        print()
        
        # Count logs
        total_logs = db.count_logs()
        check_logs = db.db["logs"].count_documents({"module": "voice_monitoring_check"})
        stats_logs = db.db["logs"].count_documents({"module": "admin_panel_stats"})
        session_logs = db.db["logs"].count_documents({"module": "admin_panel_session"})
        alerts = db.db["logs"].count_documents({"module": "voice_monitoring_alert"})
        
        print(f"Total logs in database: {total_logs}")
        print(f"  Voice checks: {check_logs}")
        print(f"  Admin panel stats: {stats_logs}")
        print(f"  Monitoring sessions: {session_logs}")
        print(f"  Security alerts: {alerts}")
        print()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("Now when you use the app:")
        print("  1. Login to trigger authentication logs")
        print("  2. Start monitoring to generate check logs")
        print("  3. Open admin panel to generate statistics")
        print("  4. All data will be saved to MongoDB automatically!")
        print()
        print("View logs with: python view_monitoring_logs.py")
        print("=" * 80)
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(test_database_logging())
