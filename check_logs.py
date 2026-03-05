"""Quick script to check what logs are in the database."""
from db import VoiceDatabase
from app.config import settings
from collections import Counter

db = VoiceDatabase(settings.MONGODB_URL)

# Get all logs
all_logs = db.get_logs(limit=500)
print(f"\n📊 Total logs in database: {len(all_logs)}")

# Count by module
modules = [log.get('module', 'unknown') for log in all_logs]
module_counts = Counter(modules)

print("\n📋 Logs by module:")
for module, count in module_counts.most_common():
    print(f"  {module}: {count}")

# Show sample logs
print("\n📝 Sample logs:")
for i, log in enumerate(all_logs[:5]):
    print(f"\n{i+1}. Module: {log.get('module')}")
    print(f"   Message: {log.get('message')}")
    print(f"   User: {log.get('user_id')}")
    print(f"   Timestamp: {log.get('timestamp')}")
