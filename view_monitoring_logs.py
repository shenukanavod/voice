"""
View Monitoring Logs
Quick script to view voice monitoring logs from the database
"""

from db import VoiceDatabase
from app.config import settings
from datetime import datetime


def view_monitoring_logs():
    """View all monitoring-related logs."""
    try:
        # Connect to database
        db = VoiceDatabase(settings.MONGODB_URL)
        
        print("=" * 80)
        print("VOICE MONITORING LOGS")
        print("=" * 80)
        print()
        
        # Get admin panel statistics
        print("📊 ADMIN PANEL STATISTICS:")
        print("-" * 80)
        stats_logs = db.get_logs(module="admin_panel_stats", limit=5)
        
        if not stats_logs:
            print("No admin panel statistics recorded yet.")
        else:
            for log in stats_logs:
                timestamp = log.get('timestamp', datetime.now())
                extra = log.get('extra_data', {})
                
                print(f"[{timestamp}]")
                print(f"  Total Sessions: {extra.get('total_sessions', 0)}")
                print(f"  Total Checks: {extra.get('total_checks', 0)}")
                print(f"  Success Rate: {extra.get('avg_success_rate', 0):.1%}")
                print(f"  Alerts: {extra.get('total_alerts', 0)}")
                print(f"  Multi-Voice Detections: {extra.get('total_multi_voice', 0)}")
                print(f"  Active Users: {', '.join(extra.get('active_users', []))}")
                print()
        
        print()
        print("=" * 80)
        print("👥 MONITORING SESSIONS:")
        print("-" * 80)
        
        # Get session logs
        session_logs = db.get_logs(module="admin_panel_session", limit=20)
        
        if not session_logs:
            print("No monitoring sessions recorded yet.")
        else:
            for log in session_logs:
                timestamp = log.get('timestamp', datetime.now())
                user_id = log.get('user_id', '-')
                message = log.get('message', '')
                extra = log.get('extra_data', {})
                
                action = extra.get('action', 'unknown')
                icon = '▶️' if action == 'session_started' else '⏹️' if action == 'session_ended' else 'ℹ️'
                
                print(f"{icon} [{timestamp}] User: {user_id}")
                print(f"   {message}")
                
                if action == 'session_started':
                    print(f"   Threshold: {extra.get('threshold', 0):.0%}, Interval: {extra.get('check_interval', 0)}s")
                elif action == 'session_ended':
                    print(f"   Total Checks: {extra.get('total_checks', 0)}")
                    print(f"   Success: {extra.get('successful_checks', 0)}, Failed: {extra.get('failed_checks', 0)}")
                    print(f"   Success Rate: {extra.get('success_rate', 0):.1%}")
                
                print()
        
        print()
        print("=" * 80)
        print("🔍 VOICE CHECKS:")
        print("-" * 80)
        check_logs = db.get_logs(module="voice_monitoring_check", limit=30)
        
        if not check_logs:
            print("No monitoring check logs found.")
        else:
            for log in check_logs:
                timestamp = log.get('timestamp', datetime.now())
                level = log.get('level', 'INFO')
                user_id = log.get('user_id', '-')
                message = log.get('message', '')
                extra = log.get('extra_data', {})
                
                icon = '✅' if level == 'INFO' else '⚠️' if level == 'WARNING' else '❌'
                
                print(f"{icon} [{timestamp}] User: {user_id}")
                print(f"   {message}")
                
                if extra:
                    if 'similarity' in extra:
                        print(f"   Similarity: {extra['similarity']:.2%}, Threshold: {extra.get('threshold', 0):.2%}")
                    if extra.get('multiple_speakers'):
                        print(f"   🚨 MULTIPLE SPEAKERS DETECTED")
                    if 'consecutive_failures' in extra and extra['consecutive_failures'] > 0:
                        print(f"   Consecutive failures: {extra['consecutive_failures']}")
                
                print()
        
        print()
        print("=" * 80)
        print("🚨 SECURITY ALERTS:")
        print("-" * 80)
        
        # Get alert logs
        alert_logs = db.get_logs(module="voice_monitoring_alert", limit=20)
        
        if not alert_logs:
            print("No security alerts found. ✅")
        else:
            for log in alert_logs:
                timestamp = log.get('timestamp', datetime.now())
                user_id = log.get('user_id', '-')
                message = log.get('message', '')
                extra = log.get('extra_data', {})
                
                print(f"🚨 [{timestamp}] User: {user_id}")
                print(f"   {message}")
                
                if extra:
                    print(f"   Consecutive failures: {extra.get('consecutive_failures', 0)}")
                    print(f"   Similarity: {extra.get('similarity', 0):.2%}")
                    if extra.get('multiple_speakers'):
                        print(f"   Multiple speakers detected: YES")
                
                print()
        
        print()
        print("=" * 80)
        print("STATISTICS:")
        print("-" * 80)
        
        # Statistics
        total_checks = db.db["logs"].count_documents({"module": "voice_monitoring_check"})
        passed_checks = db.db["logs"].count_documents({
            "module": "voice_monitoring_check",
            "level": "INFO"
        })
        failed_checks = db.db["logs"].count_documents({
            "module": "voice_monitoring_check",
            "level": "WARNING"
        })
        alerts = db.db["logs"].count_documents({"module": "voice_monitoring_alert"})
        sessions = db.db["logs"].count_documents({"module": "admin_panel_session"})
        
        print(f"Total monitoring sessions: {sessions}")
        print(f"Total voice checks: {total_checks}")
        print(f"  ✅ Passed: {passed_checks}")
        print(f"  ⚠️  Failed: {failed_checks}")
        print(f"  🚨 Security alerts: {alerts}")
        
        if total_checks > 0:
            success_rate = (passed_checks / total_checks) * 100
            print(f"\nOverall success rate: {success_rate:.1f}%")
        
        print("=" * 80)
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(view_monitoring_logs())
