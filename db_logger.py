"""
Database Logger Module
Custom logging handler that stores logs in MongoDB
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime

from db import VoiceDatabase


class MongoDBHandler(logging.Handler):
    """Custom logging handler that stores logs in MongoDB."""

    def __init__(self, db: VoiceDatabase, module_name: str = None):
        """
        Initialize MongoDB logging handler.

        Args:
            db: VoiceDatabase instance for storing logs
            module_name: Name of the module/component using this logger
        """
        super().__init__()
        self.db = db
        self.module_name = module_name

    def emit(self, record: logging.LogRecord):
        """
        Emit a log record to MongoDB.

        Args:
            record: LogRecord to be stored
        """
        try:
            # Extract user_id from extra fields if available
            user_id = getattr(record, 'user_id', None)

            # Build extra data from record
            extra_data = {
                'filename': record.filename,
                'lineno': record.lineno,
                'funcName': record.funcName,
                'process': record.process,
                'thread': record.thread,
                'threadName': record.threadName,
            }

            # Add any custom fields from extra parameter
            if hasattr(record, 'extra_fields'):
                extra_data.update(record.extra_fields)

            # Store log in database
            self.db.save_log(
                level=record.levelname,
                message=self.format(record),
                module=self.module_name or record.module,
                user_id=user_id,
                extra_data=extra_data
            )

        except Exception as e:
            # Fallback to stderr if database logging fails
            print(f"MongoDB logging error: {e}", file=sys.stderr)
            self.handleError(record)


class DatabaseLogger:
    """Wrapper class for creating loggers with database storage."""

    def __init__(self, db: VoiceDatabase, module_name: str, console_output: bool = True):
        """
        Create a logger that stores logs in database.

        Args:
            db: VoiceDatabase instance
            module_name: Name of the module/component
            console_output: Whether to also output logs to console (default: True)
        """
        self.db = db
        self.module_name = module_name
        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Add MongoDB handler
        db_handler = MongoDBHandler(db, module_name)
        db_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        db_handler.setFormatter(formatter)
        self.logger.addHandler(db_handler)

        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger

    def info(self, message: str, user_id: str = None, **kwargs):
        """Log info message."""
        self.logger.info(message, extra={'user_id': user_id, 'extra_fields': kwargs})

    def warning(self, message: str, user_id: str = None, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra={'user_id': user_id, 'extra_fields': kwargs})

    def error(self, message: str, user_id: str = None, **kwargs):
        """Log error message."""
        self.logger.error(message, extra={'user_id': user_id, 'extra_fields': kwargs})

    def debug(self, message: str, user_id: str = None, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra={'user_id': user_id, 'extra_fields': kwargs})

    def critical(self, message: str, user_id: str = None, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra={'user_id': user_id, 'extra_fields': kwargs})


def setup_database_logging(db: VoiceDatabase, module_name: str, console_output: bool = True) -> DatabaseLogger:
    """
    Convenience function to setup database logging.

    Args:
        db: VoiceDatabase instance
        module_name: Name of the module/component
        console_output: Whether to also output logs to console

    Returns:
        DatabaseLogger instance
    """
    return DatabaseLogger(db, module_name, console_output)


# Example usage and testing
if __name__ == "__main__":
    # Test the database logger
    from app.config import settings

    try:
        # Initialize database
        db = VoiceDatabase(settings.MONGODB_URL)

        # Create logger
        db_logger = setup_database_logging(db, "test_module", console_output=True)

        # Test logging
        print("Testing database logging...")
        db_logger.info("Test info message", user_id="test_user")
        db_logger.warning("Test warning message", component="audio_processing")
        db_logger.error("Test error message", error_code=500)

        print("\nRetrieving logs from database...")
        logs = db.get_logs(limit=5)

        for log in logs:
            print(f"[{log['timestamp']}] {log['level']}: {log['message']}")

        print(f"\nTotal logs in database: {db.count_logs()}")

        db.close()

    except Exception as e:
        print(f"Test failed: {e}")
