"""
DB Module
MongoDB operations for storing/retrieving voice profiles
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
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

        except ConnectionFailure as e:
            print(f"❌ MongoDB connection failed: {e}")
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

            print(f"✅ Saved embedding for user {user_id}")
            return True

        except Exception as e:
            print(f"❌ Error saving embedding: {e}")
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
            print(f"❌ Error loading embedding: {e}")
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
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ Error deleting user: {e}")
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
            self.client.close()
            print("✅ MongoDB connection closed")
