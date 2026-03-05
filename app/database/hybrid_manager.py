"""
Hybrid Database Manager for Voice Authentication System

This system intelligently switches between MongoDB Atlas (remote) 
and SQLite (local) based on availability and configuration.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Try MongoDB imports
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    AsyncIOMotorClient = None

# SQLite imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings

logger = logging.getLogger(__name__)

class HybridDatabaseManager:
    """Manages both MongoDB Atlas and SQLite databases with intelligent fallback."""
    
    def __init__(self):
        self.mongodb_client = None
        self.mongodb_db = None
        self.mongodb_connected = False
        
        self.sqlite_engine = None
        self.sqlite_session = None
        self.sqlite_connected = False
        
        self.primary_db = "none"
    
    async def initialize(self) -> str:
        """Initialize database connections. Returns primary database type."""
        
        # Try MongoDB Atlas first if configured
        if settings.MONGODB_URL and MONGODB_AVAILABLE:
            mongodb_success = await self._init_mongodb()
            if mongodb_success:
                self.primary_db = "mongodb"
                logger.info("🍃 Primary Database: MongoDB Atlas (Remote Access)")
                return "mongodb"
        
        # Fallback to SQLite
        sqlite_success = self._init_sqlite()
        if sqlite_success:
            self.primary_db = "sqlite"
            logger.info("🗃️ Primary Database: SQLite (Local)")
            return "sqlite"
        
        logger.error("❌ No database connection available!")
        return "none"
    
    async def _init_mongodb(self) -> bool:
        """Initialize MongoDB Atlas connection."""
        try:
            logger.info("🔄 Attempting MongoDB Atlas connection...")
            
            self.mongodb_client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                maxPoolSize=10
            )
            
            # Test connection
            await self.mongodb_client.admin.command('ping')
            
            self.mongodb_db = self.mongodb_client[settings.DATABASE_NAME]
            self.mongodb_connected = True
            
            logger.info("✅ MongoDB Atlas connected successfully!")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB Atlas connection failed: {str(e)}")
            self.mongodb_connected = False
            return False
    
    def _init_sqlite(self) -> bool:
        """Initialize SQLite connection."""
        try:
            logger.info("🔄 Initializing SQLite database...")
            
            # Create SQLite engine
            self.sqlite_engine = create_engine(
                settings.DATABASE_URL,
                connect_args={"check_same_thread": False},
                pool_pre_ping=True
            )
            
            # Create session factory
            SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.sqlite_engine
            )
            
            # Test connection
            with SessionLocal() as session:
                session.execute("SELECT 1")
            
            self.sqlite_session = SessionLocal
            self.sqlite_connected = True
            
            logger.info("✅ SQLite database initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ SQLite initialization failed: {str(e)}")
            self.sqlite_connected = False
            return False
    
    async def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username from primary database."""
        if self.primary_db == "mongodb" and self.mongodb_connected:
            return await self.mongodb_db.users.find_one({"username": username})
        elif self.primary_db == "sqlite" and self.sqlite_connected:
            # SQLite implementation would go here
            return {"username": username, "source": "sqlite", "mock": True}
        return None
    
    async def create_user(self, user_data: Dict[str, Any]) -> Optional[str]:
        """Create user in primary database."""
        user_data["created_at"] = datetime.now(timezone.utc)
        user_data["is_active"] = True
        user_data["is_enrolled"] = False
        
        if self.primary_db == "mongodb" and self.mongodb_connected:
            result = await self.mongodb_db.users.insert_one(user_data)
            return str(result.inserted_id)
        elif self.primary_db == "sqlite" and self.sqlite_connected:
            # SQLite implementation would go here
            return f"sqlite_user_{user_data['username']}"
        return None
    
    async def get_voice_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get voice profile for user."""
        if self.primary_db == "mongodb" and self.mongodb_connected:
            return await self.mongodb_db.voice_profiles.find_one({
                "user_id": user_id, 
                "is_active": True
            })
        elif self.primary_db == "sqlite" and self.sqlite_connected:
            # SQLite implementation would go here
            return None
        return None
    
    async def create_voice_profile(self, profile_data: Dict[str, Any]) -> Optional[str]:
        """Create voice profile."""
        profile_data["enrollment_date"] = datetime.now(timezone.utc)
        profile_data["is_active"] = True
        profile_data["verification_count"] = 0
        profile_data["success_count"] = 0
        
        if self.primary_db == "mongodb" and self.mongodb_connected:
            result = await self.mongodb_db.voice_profiles.insert_one(profile_data)
            return str(result.inserted_id)
        elif self.primary_db == "sqlite" and self.sqlite_connected:
            # SQLite implementation would go here
            return f"sqlite_profile_{profile_data['user_id']}"
        return None
    
    async def log_authentication(self, log_data: Dict[str, Any]) -> Optional[str]:
        """Log authentication attempt."""
        log_data["timestamp"] = datetime.now(timezone.utc)
        
        if self.primary_db == "mongodb" and self.mongodb_connected:
            result = await self.mongodb_db.authentication_logs.insert_one(log_data)
            return str(result.inserted_id)
        elif self.primary_db == "sqlite" and self.sqlite_connected:
            # SQLite implementation would go here
            return f"sqlite_log_{datetime.now().isoformat()}"
        return None
    
    async def get_enrolled_users(self) -> List[Dict[str, Any]]:
        """Get all enrolled users."""
        if self.primary_db == "mongodb" and self.mongodb_connected:
            cursor = self.mongodb_db.users.find({
                "is_enrolled": True, 
                "is_active": True
            })
            return await cursor.to_list(length=None)
        elif self.primary_db == "sqlite" and self.sqlite_connected:
            # SQLite implementation would return actual users
            return [
                {
                    "username": "demo_user", 
                    "source": "sqlite",
                    "enrollment_date": datetime.now(timezone.utc),
                    "is_enrolled": True
                }
            ]
        return []
    
    async def close(self):
        """Close database connections."""
        if self.mongodb_client:
            self.mongodb_client.close()
            logger.info("🍃 MongoDB Atlas connection closed")
        
        if self.sqlite_engine:
            self.sqlite_engine.dispose()
            logger.info("🗃️ SQLite connection closed")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_users': 0,
            'enrolled_users': 0,
            'voice_profiles': 0,
            'authentication_attempts': 0
        }
        
        try:
            if self.primary_db == "mongodb" and self.mongodb_connected:
                # Count users
                stats['total_users'] = await self.mongodb_db.users.count_documents({})
                stats['enrolled_users'] = await self.mongodb_db.users.count_documents({
                    "is_enrolled": True,
                    "is_active": True
                })
                
                # Count voice profiles
                stats['voice_profiles'] = await self.mongodb_db.voice_profiles.count_documents({
                    "is_active": True
                })
                
                # Count authentication logs
                stats['authentication_attempts'] = await self.mongodb_db.authentication_logs.count_documents({})
                
            elif self.primary_db == "sqlite" and self.sqlite_connected:
                # SQLite stats would be fetched from simple_db
                # For now return mock data
                stats = {
                    'total_users': 0,
                    'enrolled_users': 0,
                    'voice_profiles': 0,
                    'authentication_attempts': 0
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current database status."""
        return {
            "primary_database": self.primary_db,
            "mongodb_connected": self.mongodb_connected,
            "sqlite_connected": self.sqlite_connected,
            "remote_access": self.mongodb_connected,
            "multi_computer_support": self.mongodb_connected
        }

# Global database manager
_db_manager: Optional[HybridDatabaseManager] = None

async def get_database_manager() -> HybridDatabaseManager:
    """Get or create database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = HybridDatabaseManager()
        await _db_manager.initialize()
    
    return _db_manager

async def close_database_manager():
    """Close database manager."""
    global _db_manager
    
    if _db_manager:
        await _db_manager.close()
        _db_manager = None