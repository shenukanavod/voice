# Database Logging System

All logs from the voice authentication system are now stored in MongoDB for centralized logging, monitoring, and analysis.

## Features

- **Centralized Logging**: All application logs stored in MongoDB
- **User Tracking**: Logs can be filtered by user_id
- **Module Tracking**: Logs categorized by module (api, desktop_app, voice_monitoring, etc.)
- **Log Levels**: Support for DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Extra Metadata**: Additional contextual data stored with each log
- **Automatic Cleanup**: Old logs can be automatically deleted
- **Log Viewer**: Built-in CLI tool for viewing and analyzing logs

## Database Structure

Logs are stored in the `logs` collection with the following structure:

```json
{
  "timestamp": "2026-02-26T10:30:45.123Z",
  "level": "INFO",
  "message": "User authenticated successfully",
  "module": "api",
  "user_id": "john_doe",
  "extra_data": {
    "similarity": 0.95,
    "threshold": 0.90,
    "filename": "api.py",
    "lineno": 123
  }
}
```

## Usage

### Viewing Logs

Use the `log_viewer.py` command-line tool:

```bash
# View recent logs (default: 50)
python log_viewer.py

# View more logs
python log_viewer.py --recent 100

# View logs for specific user
python log_viewer.py --user john_doe

# View only errors
python log_viewer.py --errors

# View logs by level
python log_viewer.py --level ERROR --limit 25

# View statistics
python log_viewer.py --stats

# Clean up old logs (older than 30 days)
python log_viewer.py --cleanup 30
```

### In Your Code

#### Using the Database Logger

```python
from db import VoiceDatabase
from db_logger import setup_database_logging
from app.config import settings

# Initialize database
db = VoiceDatabase(settings.MONGODB_URL)

# Setup logger
db_logger = setup_database_logging(db, "my_module", console_output=True)

# Log with user context
db_logger.info("User logged in", user_id="john_doe")
db_logger.warning("Failed attempt", user_id="jane_doe", attempts=3)
db_logger.error("Authentication failed", user_id="bob", error_code=401)
```

#### Direct Database Logging

```python
from db import VoiceDatabase
from app.config import settings

db = VoiceDatabase(settings.MONGODB_URL)

# Save log directly
db.save_log(
    level="INFO",
    message="User registered successfully",
    module="enrollment",
    user_id="alice",
    extra_data={"samples_recorded": 3}
)
```

## Logged Modules

The following modules are configured to log to the database:

- **api**: REST API operations
- **desktop_app**: Desktop GUI application
- **voice_monitoring**: Real-time voice monitoring
- **monitoring_gui**: Monitoring window interface
- **db**: Database operations
- **enrollment**: User enrollment operations
- **verification**: Authentication attempts

## Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical errors requiring immediate attention

## Querying Logs

You can query logs directly using MongoDB:

```python
from db import VoiceDatabase
from app.config import settings

db = VoiceDatabase(settings.MONGODB_URL)

# Get recent logs
logs = db.get_logs(limit=100)

# Get logs for specific user
user_logs = db.get_logs(user_id="john_doe", limit=50)

# Get error logs
errors = db.get_logs(level="ERROR", limit=25)

# Get logs from specific module
api_logs = db.get_logs(module="api", limit=50)

# Count logs
total = db.count_logs()
errors = db.count_logs(level="ERROR")
```

## Maintenance

### Auto-Cleanup

To prevent the logs collection from growing too large, periodically clean up old logs:

```python
from db import VoiceDatabase
from app.config import settings

db = VoiceDatabase(settings.MONGODB_URL)

# Delete logs older than 30 days
deleted = db.delete_old_logs(days=30)
print(f"Deleted {deleted} old logs")
```

### Scheduled Cleanup

You can set up a scheduled task (cron job on Linux, Task Scheduler on Windows) to run:

```bash
python -c "from db import VoiceDatabase; from app.config import settings; db = VoiceDatabase(settings.MONGODB_CONNECTION_STRING); db.delete_old_logs(30)"
```

## Monitoring and Alerts

All critical events are logged with level `CRITICAL`:

- Multiple failed authentication attempts
- Voice monitoring security alerts
- System errors

You can set up monitoring to alert on CRITICAL logs:

```python
# Get critical logs from the last hour
from datetime import datetime, timedelta
from db import VoiceDatabase

db = VoiceDatabase(settings.MONGODB_URL)
collection = db.db["logs"]

one_hour_ago = datetime.utcnow() - timedelta(hours=1)
critical_logs = collection.find({
    "level": "CRITICAL",
    "timestamp": {"$gte": one_hour_ago}
})

for log in critical_logs:
    # Send alert notification
    print(f"ALERT: {log['message']}")
```

## Performance Considerations

- Logs are written asynchronously and won't block application execution
- Failed database writes fall back to console output
- Indexes can be added to improve query performance:

```javascript
// In MongoDB shell
use voice_auth_db
db.logs.createIndex({ "timestamp": -1 })
db.logs.createIndex({ "user_id": 1, "timestamp": -1 })
db.logs.createIndex({ "level": 1, "timestamp": -1 })
db.logs.createIndex({ "module": 1, "timestamp": -1 })
```

## Troubleshooting

If logs are not appearing in the database:

1. Check MongoDB connection string in `app/config.py`
2. Verify MongoDB is running and accessible
3. Check console output for database connection errors
4. Ensure the database user has write permissions
5. Review console logs - they continue even if database logging fails

## Benefits

- **Centralized**: All logs in one place
- **Searchable**: Query by user, module, level, time range
- **Persistent**: Logs survive application restarts
- **Analyzable**: Use MongoDB aggregation for insights
- **Auditable**: Complete audit trail of all operations
- **Scalable**: MongoDB can handle large volumes of logs
