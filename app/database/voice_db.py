"""
Database Module - MongoDB storage for voice embeddings.

Stores only:
- user_id: Unique identifier
- voice_embedding: 128-dimensional array
- created_at: Timestamp
- updated_at: Timestamp

NEVER stores:
- Raw audio
- Audio files
- Temporary data
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import certifi
import numpy as np
from pymongo import MongoClient


class VoiceDatabase:
    """
    MongoDB database for storing voice embeddings.
    """

    def __init__(
        self, connection_string: Optional[str] = None, database_name: str = "voice_auth"
    ):
        """
        Initialize database connection.

        Args:
            connection_string: MongoDB connection string (if None, uses localhost)
            database_name: Database name
        """
        # Use provided connection string or default to localhost
        if connection_string is None:
            connection_string = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")

        try:
            # Use certifi CA bundle for proper certificate chain
            self.client = MongoClient(connection_string, tlsCAFile=certifi.where())
            self.db = self.client[database_name]
            self.collection = self.db["voice_profiles"]

            # Create index on user_id for fast lookups (skip if already exists)
            try:
                self.collection.create_index("user_id", unique=True)
            except Exception as index_error:
                # Index already exists, ignore the error
                pass

            print(f"✅ Connected to MongoDB: {database_name}")
        except Exception as e:
            error_str = str(e)
            if (
                "SSL" in error_str
                or "CERTIFICATE" in error_str
                or "certificate" in error_str
            ):
                print(f"⚠️ MongoDB SSL Error - retrying with SSL bypass...")
                # Retry with SSL certificate verification disabled
                if "tlsAllowInvalidCertificates=true" not in connection_string:
                    connection_string = (
                        connection_string.replace("?", "&")
                        if "?" in connection_string
                        else connection_string + "?"
                    )
                    if "?" not in connection_string:
                        connection_string += "?tlsAllowInvalidCertificates=true"
                    elif "&tlsAllowInvalidCertificates=true" not in connection_string:
                        connection_string += "&tlsAllowInvalidCertificates=true"

                try:
                    self.client = MongoClient(
                        connection_string,
                        tlsAllowInvalidCertificates=True,
                        tlsCAFile=certifi.where(),
                        serverSelectionTimeoutMS=5000,
                    )
                    self.db = self.client[database_name]
                    self.collection = self.db["voice_profiles"]
                    print(f"✅ Connected to MongoDB with SSL bypass: {database_name}")
                except Exception as retry_error:
                    print(f"❌ MongoDB connection failed: {retry_error}")
                    raise Exception(f"MongoDB SSL Error: {error_str}")
            else:
                print(f"❌ MongoDB connection failed: {e}")
                raise

    def save_embedding(self, user_id: str, embedding: np.ndarray) -> bool:
        """
        Save or update voice embedding for a user.


        """
        try:
            # Validate embedding (flexible size now)
            if len(embedding.shape) != 1:
                raise ValueError(
                    f"Embedding must be 1-dimensional, got shape {embedding.shape}"
                )

            print(f" Saving embedding: {len(embedding)} dimensions for user {user_id}")

            # Convert numpy array to list for MongoDB storage
            embedding_list = embedding.tolist()

            # Create document
            document = {
                "user_id": user_id,
                "voice_embedding": embedding_list,
                "updated_at": datetime.utcnow(),
            }

            # Upsert (update if exists, insert if not)
            result = self.collection.update_one(
                {"user_id": user_id},
                {"$set": document, "$setOnInsert": {"created_at": datetime.utcnow()}},
                upsert=True,
            )

            if result.upserted_id:
                print(f" New enrollment saved for user: {user_id}")
            else:
                print(f"Embedding updated for user: {user_id}")

            return True

        except Exception as e:
            print(f" Failed to save embedding: {e}")
            return False

    def get_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Retrieve voice embedding for a user.

        """
        try:
            document = self.collection.find_one({"user_id": user_id})

            if document is None:
                print(f" No embedding found for user: {user_id}")
                return None

            # Convert list back to numpy array
            embedding = np.array(document["voice_embedding"], dtype=np.float32)

            print(f" Retrieved embedding for user: {user_id}")
            return embedding

        except Exception as e:
            print(f" Failed to retrieve embedding: {e}")
            return None

    def user_exists(self, user_id: str) -> bool:
        """
        Check if user is enrolled.


        """
        count = self.collection.count_documents({"user_id": user_id})
        return count > 0

    def delete_user(self, user_id: str) -> bool:
        """
        Delete user enrollment.

        """
        try:
            result = self.collection.delete_one({"user_id": user_id})

            if result.deleted_count > 0:
                print(f" Deleted user: {user_id}")
                return True
            else:
                print(f" User not found: {user_id}")
                return False

        except Exception as e:
            print(f" Failed to delete user: {e}")
            return False

    def get_all_users(self) -> List[str]:
        """
        Get list of all enrolled users.

        """
        try:
            users = self.collection.distinct("user_id")
            return users
        except Exception as e:
            print(f"❌ Failed to retrieve users: {e}")
            return []

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Get user enrollment information.


        """
        try:
            document = self.collection.find_one({"user_id": user_id})

            if document is None:
                return None

            return {
                "user_id": document["user_id"],
                "created_at": document.get("created_at"),
                "updated_at": document.get("updated_at"),
                "embedding_size": len(document["voice_embedding"]),
            }

        except Exception as e:
            print(f" Failed to retrieve user info: {e}")
            return None

    def count_users(self) -> int:
        """
        Count total enrolled users.

        """
        return self.collection.count_documents({})

    def close(self):
        """Close database connection."""
        self.client.close()
        print(" Database connection closed")


def test_database():
    """Test database operations."""
    print("\n" + "=" * 60)
    print("DATABASE TEST")
    print("=" * 60 + "\n")

    # Initialize database
    db = VoiceDatabase(database_name="voice_auth_test")

    try:
        # Test 1: Save embedding
        print("TEST 1: Save embedding")
        test_user = "test_user_001"
        test_embedding = np.random.randn(128).astype(np.float32)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)

        success = db.save_embedding(test_user, test_embedding)
        print(f"   Result: {' Success' if success else ' Failed'}\n")

        # Test 2: Check if user exists
        print("TEST 2: Check user exists")
        exists = db.user_exists(test_user)
        print(f"   User exists: {exists}\n")

        # Test 3: Retrieve embedding
        print("TEST 3: Retrieve embedding")
        retrieved_embedding = db.get_embedding(test_user)
        if retrieved_embedding is not None:
            print(f"   Retrieved shape: {retrieved_embedding.shape}")
            print(
                f"   Match original: {np.allclose(test_embedding, retrieved_embedding)}\n"
            )

        # Test 4: Get user info
        print("TEST 4: Get user info")
        info = db.get_user_info(test_user)
        if info:
            print(f"   User ID: {info['user_id']}")
            print(f"   Created: {info['created_at']}")
            print(f"   Embedding size: {info['embedding_size']}\n")

        # Test 5: Count users
        print("TEST 5: Count users")
        count = db.count_users()
        print(f"   Total users: {count}\n")

        # Test 6: Delete user
        print("TEST 6: Delete user")
        deleted = db.delete_user(test_user)
        print(f"   Deleted: {deleted}")
        exists_after = db.user_exists(test_user)
        print(f"   Still exists: {exists_after}\n")

        print(" All database tests completed!")

    finally:
        db.close()


if __name__ == "__main__":
    test_database()
    os.system("python speaker_verification/train_gpu.py")
