"""
Database Models for Voice Authentication System

This module defines SQLAlchemy models for storing user data, authentication logs,
monitoring sessions, and security alerts. Voice profiles remain encrypted files
but are tracked in the database for better management.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import text
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()

class User(Base):
    """User model for storing basic user information."""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    full_name = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_enrolled = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_enrolled': self.is_enrolled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
        }

class VoiceProfile(Base):
    """Voice profile model for tracking encrypted voice profile files."""
    __tablename__ = "voice_profiles"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    profile_hash = Column(String(64), nullable=False)  # SHA-256 hash of encrypted file
    file_path = Column(String(500), nullable=False)  # Path to encrypted profile file
    file_size = Column(Integer, nullable=True)  # File size in bytes
    enrollment_date = Column(DateTime(timezone=True), server_default=func.now())
    last_verified = Column(DateTime(timezone=True), nullable=True)
    verification_count = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    metadata = Column(JSON, nullable=True)  # Additional metadata as JSON
    
    def __repr__(self):
        return f"<VoiceProfile(id='{self.id}', user_id='{self.user_id}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voice profile to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'profile_hash': self.profile_hash,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'enrollment_date': self.enrollment_date.isoformat() if self.enrollment_date else None,
            'last_verified': self.last_verified.isoformat() if self.last_verified else None,
            'verification_count': self.verification_count,
            'success_count': self.success_count,
            'is_active': self.is_active,
            'metadata': self.metadata,
        }

class AuthenticationLog(Base):
    """Authentication log model for tracking all authentication attempts."""
    __tablename__ = "authentication_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    authentication_type = Column(String(50), nullable=False)  # 'enrollment', 'verification', 'quick_verify'
    success = Column(Boolean, nullable=False)
    similarity_score = Column(Float, nullable=True)
    threshold_used = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    error_message = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<AuthenticationLog(id='{self.id}', user_id='{self.user_id}', success={self.success})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert authentication log to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'authentication_type': self.authentication_type,
            'success': self.success,
            'similarity_score': self.similarity_score,
            'threshold_used': self.threshold_used,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }

class MonitoringSession(Base):
    """Monitoring session model for continuous monitoring sessions."""
    __tablename__ = "monitoring_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_name = Column(String(200), nullable=False)
    authorized_user_id = Column(String(36), nullable=False, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    alert_count = Column(Integer, default=0, nullable=False)
    total_audio_processed = Column(Integer, default=0, nullable=False)  # Total audio chunks processed
    configuration = Column(JSON, nullable=True)  # Session configuration as JSON
    
    def __repr__(self):
        return f"<MonitoringSession(id='{self.id}', name='{self.session_name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert monitoring session to dictionary."""
        return {
            'id': self.id,
            'session_name': self.session_name,
            'authorized_user_id': self.authorized_user_id,
            'description': self.description,
            'is_active': self.is_active,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'alert_count': self.alert_count,
            'total_audio_processed': self.total_audio_processed,
            'configuration': self.configuration,
        }

class SecurityAlert(Base):
    """Security alert model for storing security alerts from monitoring sessions."""
    __tablename__ = "security_alerts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), nullable=False, index=True)
    alert_type = Column(String(100), nullable=False, index=True)  # 'unauthorized_speaker', 'spoofing_detected', 'voice_change'
    severity = Column(String(20), nullable=False, index=True)  # 'low', 'medium', 'high', 'critical'
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)  # Confidence in the alert (0.0 - 1.0)
    detected_user_id = Column(String(36), nullable=True)  # If a user was detected
    audio_segment_path = Column(String(500), nullable=True)  # Path to audio segment that triggered alert
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String(36), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved = Column(Boolean, default=False, nullable=False)
    resolution_notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<SecurityAlert(id='{self.id}', type='{self.alert_type}', severity='{self.severity}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security alert to dictionary."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'confidence_score': self.confidence_score,
            'detected_user_id': self.detected_user_id,
            'audio_segment_path': self.audio_segment_path,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved': self.resolved,
            'resolution_notes': self.resolution_notes,
        }

class SystemMetrics(Base):
    """System metrics model for storing performance and usage statistics."""
    __tablename__ = "system_metrics"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_type = Column(String(100), nullable=False, index=True)  # 'performance', 'usage', 'error'
    metric_name = Column(String(200), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)  # 'seconds', 'count', 'percentage', etc.
    tags = Column(JSON, nullable=True)  # Additional tags as JSON
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.metric_value})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system metrics to dictionary."""
        return {
            'id': self.id,
            'metric_type': self.metric_type,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'unit': self.unit,
            'tags': self.tags,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }

# Create indexes for better performance
def create_indexes(engine):
    """Create additional indexes for better query performance."""
    with engine.connect() as conn:
        # Composite indexes for common queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_auth_logs_user_timestamp 
            ON authentication_logs(user_id, timestamp DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_auth_logs_success_timestamp 
            ON authentication_logs(success, timestamp DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_alerts_session_timestamp 
            ON security_alerts(session_id, timestamp DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_alerts_severity_timestamp 
            ON security_alerts(severity, timestamp DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_metrics_type_name_timestamp 
            ON system_metrics(metric_type, metric_name, timestamp DESC)
        """))
        
        conn.commit()
