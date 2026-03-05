"""
DB Module
MongoDB operations for storing/retrieving voice profiles and logs
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class VoiceDatabase:
    """MongoDB wrapper for voice profile storage."""

    def __init__(self, connection_string: str, database_name: str = "voice_auth_db"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None

        self._connect()

    def _connect(self):
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.database_name]

            # Test connection
            self.db.command("ping")
            print(f"✅ Connected to MongoDB: {self.database_name}")
            
            # Log connection (after connection is established)
            try:
                self.save_log("INFO", f"Connected to MongoDB: {self.database_name}", module="db")
            except:
                pass  # Ignore if logging fails during initial connection

        except ConnectionFailure as e:
            error_msg = f"MongoDB connection failed: {e}"
            print(f"❌ {error_msg}")
            raise Exception(f"Cannot connect to MongoDB: {e}")

    def save_embedding(self, user_id: str, embedding: np.ndarray) -> bool:
        """
        Save speaker embedding to MongoDB.

        Args:
            user_id: Unique user identifier
            embedding: 128-D embedding vector

        Returns:
            True if successful
        """
        try:
            collection = self.db["voice_profiles"]

            embedding_data = {
                "user_id": user_id,
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding),
            }

            # Upsert
            collection.update_one({"user_id": user_id}, {"$set": embedding_data}, upsert=True)

            msg = f"Saved embedding for user {user_id}"
            print(f"✅ {msg}")
            self.save_log("INFO", msg, module="db", user_id=user_id)
            return True

        except Exception as e:
            error_msg = f"Error saving embedding: {e}"
            print(f"❌ {error_msg}")
            self.save_log("ERROR", error_msg, module="db", user_id=user_id)
            return False

    def get_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Load speaker embedding from MongoDB.

        Args:
            user_id: User identifier

        Returns:
            128-D embedding vector or None
        """
        try:
            collection = self.db["voice_profiles"]
            doc = collection.find_one({"user_id": user_id})

            if doc and "embedding" in doc:
                return np.array(doc["embedding"], dtype=np.float32)

            return None

        except Exception as e:
            error_msg = f"Error loading embedding: {e}"
            print(f"❌ {error_msg}")
            # Don't log to database here to avoid recursion
            return None

    def user_exists(self, user_id: str) -> bool:
        """Check if user profile exists."""
        try:
            collection = self.db["voice_profiles"]
            return collection.find_one({"user_id": user_id}) is not None
        except Exception as e:
            print(f"❌ Error checking user: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """Delete user profile."""
        try:
            collection = self.db["voice_profiles"]
            result = collection.delete_one({"user_id": user_id})
            if result.deleted_count > 0:
                self.save_log("INFO", f"Deleted user profile", module="db", user_id=user_id)
            return result.deleted_count > 0
        except Exception as e:
            error_msg = f"Error deleting user: {e}"
            print(f"❌ {error_msg}")
            self.save_log("ERROR", error_msg, module="db", user_id=user_id)
            return False

    def get_all_users(self) -> List[str]:
        """Get list of all enrolled users."""
        try:
            collection = self.db["voice_profiles"]
            docs = collection.find({}, {"user_id": 1})
            return [doc["user_id"] for doc in docs]
        except Exception as e:
            print(f"❌ Error listing users: {e}")
            return []

    def count_users(self) -> int:
        """Count enrolled users."""
        try:
            collection = self.db["voice_profiles"]
            return collection.count_documents({})
        except Exception as e:
            print(f"❌ Error counting users: {e}")
            return 0

    def close(self):
        """Close database connection."""
        if self.client:
            try:
                self.save_log("INFO", "MongoDB connection closed", module="db")
            except:
                pass  # Ignore if logging fails during shutdown
            self.client.close()
            print("✅ MongoDB connection closed")

    def save_log(
        self,
        level: str,
        message: str,
        module: str = None,
        user_id: str = None,
        extra_data: Dict[str, Any] = None
    ) -> bool:
        """
        Save log entry to MongoDB.

        Args:
            level: Log level (INFO, WARNING, ERROR, DEBUG, etc.)
            message: Log message
            module: Module/file name where log originated
            user_id: Associated user ID (if applicable)
            extra_data: Additional metadata dictionary

        Returns:
            True if successful
        """
        try:
            collection = self.db["logs"]

            log_entry = {
                "timestamp": datetime.utcnow(),
                "level": level,
                "message": message,
                "module": module,
                "user_id": user_id,
                "extra_data": extra_data or {}
            }

            collection.insert_one(log_entry)
            return True

        except Exception as e:
            # Fallback to print if database logging fails
            print(f"❌ Error saving log to database: {e}")
            print(f"   Original log: [{level}] {message}")
            return False

    def get_logs(
        self,
        level: str = None,
        user_id: str = None,
        module: str = None,
        limit: int = 100,
        skip: int = 0,
        custom_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs from MongoDB.

        Args:
            level: Filter by log level
            user_id: Filter by user ID
            module: Filter by module name
            limit: Maximum number of logs to return
            skip: Number of logs to skip (for pagination)
            custom_filter: Additional MongoDB filter criteria

        Returns:
            List of log documents
        """
        try:
            collection = self.db["logs"]

            # Build filter query
            query = {}
            if level:
                query["level"] = level
            if user_id:
                query["user_id"] = user_id
            if module:
                query["module"] = module
            
            # Merge custom filter if provided
            if custom_filter:
                query.update(custom_filter)

            # Retrieve logs sorted by timestamp (newest first)
            logs = collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)

            return list(logs)

        except Exception as e:
            print(f"❌ Error retrieving logs: {e}")
            return []

    def delete_old_logs(self, days: int = 30) -> int:
        """
        Delete logs older than specified days.

        Args:
            days: Delete logs older than this many days

        Returns:
            Number of logs deleted
        """
        try:
            from datetime import timedelta
            collection = self.db["logs"]

            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = collection.delete_many({"timestamp": {"$lt": cutoff_date}})

            deleted_count = result.deleted_count
            if deleted_count > 0:
                print(f"✅ Deleted {deleted_count} old logs (older than {days} days)")

            return deleted_count

        except Exception as e:
            print(f"❌ Error deleting old logs: {e}")
            return 0

    def count_logs(self, level: str = None) -> int:
        """
        Count logs, optionally filtered by level.

        Args:
            level: Filter by log level (optional)

        Returns:
            Number of logs
        """
        try:
            collection = self.db["logs"]
            query = {"level": level} if level else {}
            return collection.count_documents(query)

        except Exception as e:
            print(f"❌ Error counting logs: {e}")
            return 0

