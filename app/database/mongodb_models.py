"""
MongoDB Models for Voice Authentication System

This module defines MongoDB document models using Motor (async MongoDB driver)
for the voice authentication system.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, DESCENDING
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field
from bson import ObjectId

class PyObjectId(ObjectId):
    """Custom ObjectId class for Pydantic compatibility."""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class MongoBaseModel(BaseModel):
    """Base model for MongoDB documents."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserDocument(MongoBaseModel):
    """User document model for MongoDB."""
    username: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = Field(None, max_length=255)
    full_name: Optional[str] = Field(None, max_length=200)
    is_active: bool = Field(default=True)
    is_enrolled: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    
    # Voice profile information
    voice_profile: Optional[Dict[str, Any]] = Field(default=None)
    
    # User metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class VoiceProfileDocument(MongoBaseModel):
    """Voice profile document model for MongoDB."""
    user_id: str = Field(..., min_length=1)
    profile_hash: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    file_size: Optional[int] = None
    enrollment_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified: Optional[datetime] = None
    verification_count: int = Field(default=0)
    success_count: int = Field(default=0)
    is_active: bool = Field(default=True)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class AuthenticationLogDocument(MongoBaseModel):
    """Authentication log document model for MongoDB."""
    user_id: str = Field(..., min_length=1)
    authentication_type: str = Field(..., min_length=1)  # 'enrollment', 'verification', 'quick_verify'
    success: bool
    similarity_score: Optional[float] = None
    threshold_used: Optional[float] = None
    processing_time: Optional[float] = None  # Processing time in seconds
    error_message: Optional[str] = None
    ip_address: Optional[str] = Field(None, max_length=45)
    user_agent: Optional[str] = None
    session_id: Optional[str] = Field(None, max_length=100)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Additional context
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MonitoringSessionDocument(MongoBaseModel):
    """Monitoring session document model for MongoDB."""
    session_name: str = Field(..., min_length=1, max_length=200)
    authorized_user_id: str = Field(..., min_length=1)
    description: Optional[str] = None
    is_active: bool = Field(default=True)
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    alert_count: int = Field(default=0)
    total_audio_processed: int = Field(default=0)
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SecurityAlertDocument(MongoBaseModel):
    """Security alert document model for MongoDB."""
    session_id: str = Field(..., min_length=1)
    alert_type: str = Field(..., min_length=1)  # 'unauthorized_speaker', 'spoofing_detected', etc.
    severity: str = Field(..., regex="^(low|medium|high|critical)$")
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    detected_user_id: Optional[str] = None
    audio_segment_path: Optional[str] = Field(None, max_length=500)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = Field(default=False)
    resolution_notes: Optional[str] = None
    
    # Additional context and metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SystemMetricsDocument(MongoBaseModel):
    """System metrics document model for MongoDB."""
    metric_type: str = Field(..., min_length=1)  # 'performance', 'usage', 'error'
    metric_name: str = Field(..., min_length=1, max_length=200)
    metric_value: float
    unit: Optional[str] = Field(None, max_length=50)
    tags: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MongoDBManager:
    """MongoDB connection and operations manager."""
    
    def __init__(self, connection_string: str, database_name: str):
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(connection_string)
        self.db: AsyncIOMotorDatabase = self.client[database_name]
        
        # Collections
        self.users: AsyncIOMotorCollection = self.db.users
        self.voice_profiles: AsyncIOMotorCollection = self.db.voice_profiles
        self.auth_logs: AsyncIOMotorCollection = self.db.authentication_logs
        self.monitoring_sessions: AsyncIOMotorCollection = self.db.monitoring_sessions
        self.security_alerts: AsyncIOMotorCollection = self.db.security_alerts
        self.system_metrics: AsyncIOMotorCollection = self.db.system_metrics
    
    async def create_indexes(self):
        """Create database indexes for optimal performance."""
        
        # Users collection indexes
        await self.users.create_indexes([
            IndexModel([("username", ASCENDING)], unique=True),
            IndexModel([("email", ASCENDING)], unique=True, sparse=True),
            IndexModel([("is_active", ASCENDING)]),
            IndexModel([("is_enrolled", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)])
        ])
        
        # Voice profiles collection indexes
        await self.voice_profiles.create_indexes([
            IndexModel([("user_id", ASCENDING)], unique=True),
            IndexModel([("is_active", ASCENDING)]),
            IndexModel([("enrollment_date", DESCENDING)]),
            IndexModel([("last_verified", DESCENDING)])
        ])
        
        # Authentication logs collection indexes
        await self.auth_logs.create_indexes([
            IndexModel([("user_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("success", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("authentication_type", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("session_id", ASCENDING)], sparse=True)
        ])
        
        # Monitoring sessions collection indexes
        await self.monitoring_sessions.create_indexes([
            IndexModel([("authorized_user_id", ASCENDING)]),
            IndexModel([("is_active", ASCENDING)]),
            IndexModel([("start_time", DESCENDING)]),
            IndexModel([("session_name", ASCENDING)])
        ])
        
        # Security alerts collection indexes
        await self.security_alerts.create_indexes([
            IndexModel([("session_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("severity", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("alert_type", ASCENDING)]),
            IndexModel([("acknowledged", ASCENDING)]),
            IndexModel([("resolved", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)])
        ])
        
        # System metrics collection indexes
        await self.system_metrics.create_indexes([
            IndexModel([("metric_type", ASCENDING), ("metric_name", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("metric_type", ASCENDING)])
        ])
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            # Test connection
            await self.client.admin.command('ping')
            
            # Get database stats
            stats = await self.db.command("dbStats")
            
            return {
                "status": "healthy",
                "database": self.db.name,
                "collections": stats.get("collections", 0),
                "data_size": stats.get("dataSize", 0),
                "index_size": stats.get("indexSize", 0),
                "storage_size": stats.get("storageSize", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self):
        """Close database connection."""
        self.client.close()

# Global MongoDB manager instance
mongodb_manager: Optional[MongoDBManager] = None

def get_mongodb_manager(connection_string: str, database_name: str) -> MongoDBManager:
    """Get MongoDB manager instance."""
    global mongodb_manager
    if mongodb_manager is None:
        mongodb_manager = MongoDBManager(connection_string, database_name)
    return mongodb_manager

async def init_mongodb(connection_string: str, database_name: str) -> MongoDBManager:
    """Initialize MongoDB connection and create indexes."""
    manager = get_mongodb_manager(connection_string, database_name)
    await manager.create_indexes()
    return manager
