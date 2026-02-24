"""
MongoDB Database Manager for Voice Authentication System

This module handles MongoDB Atlas connection and operations,
replacing SQLite for remote multi-computer access.
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import os

logger = logging.getLogger(__name__)

class MongoDBManager:
    """MongoDB Atlas connection manager."""
    
    def __init__(self, connection_url: str, database_name: str):
        self.connection_url = connection_url
        self.database_name = database_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self._is_connected = False
    
    async def connect(self) -> bool:
        """Connect to MongoDB Atlas."""
        try:
            logger.info("Connecting to MongoDB Atlas...")
            self.client = AsyncIOMotorClient(
                self.connection_url,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test the connection
            await self.client.admin.command('ping')
            
            self.database = self.client[self.database_name]
            self._is_connected = True
            
            logger.info(f"✅ Successfully connected to MongoDB Atlas!")
            logger.info(f"📊 Database: {self.database_name}")
            
            # Create indexes for better performance
            await self._create_indexes()
            
            return True
            
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to MongoDB: {str(e)}")
            self._is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("📊 Disconnected from MongoDB Atlas")
    
    async def _create_indexes(self):
        """Create database indexes for better performance."""
        try:
            # Users collection indexes
            await self.database.users.create_index("username", unique=True)
            await self.database.users.create_index("email", unique=True, sparse=True)
            
            # Voice profiles indexes
            await self.database.voice_profiles.create_index("user_id")
            await self.database.voice_profiles.create_index("enrollment_date")
            
            # Authentication logs indexes
            await self.database.authentication_logs.create_index("user_id")
            await self.database.authentication_logs.create_index("timestamp")
            await self.database.authentication_logs.create_index("authentication_type")
            
            logger.info("📊 Database indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not create indexes: {str(e)}")
    
    async def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        if not self._is_connected:
            return None
        return await self.database.users.find_one({"username": username})
    
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create new user."""
        user_data["created_at"] = datetime.now(timezone.utc)
        user_data["is_active"] = True
        user_data["is_enrolled"] = False
        
        result = await self.database.users.insert_one(user_data)
        return str(result.inserted_id)
    
    async def get_voice_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get voice profile for user."""
        if not self._is_connected:
            return None
        return await self.database.voice_profiles.find_one({"user_id": user_id, "is_active": True})
    
    async def create_voice_profile(self, profile_data: Dict[str, Any]) -> str:
        """Create voice profile."""
        profile_data["enrollment_date"] = datetime.now(timezone.utc)
        profile_data["verification_count"] = 0
        profile_data["success_count"] = 0
        profile_data["is_active"] = True
        
        result = await self.database.voice_profiles.insert_one(profile_data)
        return str(result.inserted_id)
    
    async def log_authentication(self, log_data: Dict[str, Any]) -> str:
        """Log authentication attempt."""
        log_data["timestamp"] = datetime.now(timezone.utc)
        result = await self.database.authentication_logs.insert_one(log_data)
        return str(result.inserted_id)
    
    async def get_enrolled_users(self) -> List[Dict[str, Any]]:
        """Get all enrolled users."""
        if not self._is_connected:
            return []
        
        cursor = self.database.users.find({"is_enrolled": True, "is_active": True})
        return await cursor.to_list(length=None)
    
    async def update_user_login(self, username: str):
        """Update user's last login time."""
        await self.database.users.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.now(timezone.utc)}}
        )
    
    async def get_authentication_stats(self, user_id: str) -> Dict[str, Any]:
        """Get authentication statistics for user."""
        if not self._is_connected:
            return {}
        
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$success",
                "count": {"$sum": 1},
                "avg_score": {"$avg": "$similarity_score"}
            }}
        ]
        
        cursor = self.database.authentication_logs.aggregate(pipeline)
        results = await cursor.to_list(length=None)
        
        stats = {"total_attempts": 0, "successful_attempts": 0, "failed_attempts": 0}
        for result in results:
            if result["_id"]:  # Success = True
                stats["successful_attempts"] = result["count"]
                stats["avg_success_score"] = result.get("avg_score", 0)
            else:  # Success = False
                stats["failed_attempts"] = result["count"]
        
        stats["total_attempts"] = stats["successful_attempts"] + stats["failed_attempts"]
        return stats

    def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        return self._is_connected

# Global MongoDB manager instance
_mongo_manager: Optional[MongoDBManager] = None

def get_mongodb_manager() -> Optional[MongoDBManager]:
    """Get MongoDB manager instance."""
    return _mongo_manager

async def initialize_mongodb(connection_url: str, database_name: str) -> bool:
    """Initialize MongoDB connection."""
    global _mongo_manager
    
    _mongo_manager = MongoDBManager(connection_url, database_name)
    success = await _mongo_manager.connect()
    
    if success:
        logger.info("🗃️ MongoDB Atlas is now the primary database")
    else:
        logger.error("❌ MongoDB Atlas connection failed")
    
    return success

async def close_mongodb():
    """Close MongoDB connection."""
    global _mongo_manager
    
    if _mongo_manager:
        await _mongo_manager.disconnect()
        _mongo_manager = None