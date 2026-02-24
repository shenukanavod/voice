"""
Database Models for Exam Lockdown Browser System

This module defines specialized database models for an examination system
with voice authentication and continuous monitoring capabilities.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import text, Index
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

Base = declarative_base()

class ExamStatus(str, Enum):
    """Exam status enumeration."""
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"

class SessionStatus(str, Enum):
    """Exam session status enumeration."""
    PENDING = "pending"
    AUTHENTICATING = "authenticating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

class AlertSeverity(str, Enum):
    """Security alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Student(Base):
    """Student model for exam system."""
    __tablename__ = "students"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String(50), unique=True, nullable=False, index=True)  # University ID
    email = Column(String(255), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    full_name = Column(String(200), nullable=False)
    
    # Voice authentication
    is_voice_enrolled = Column(Boolean, default=False, nullable=False)
    voice_enrollment_date = Column(DateTime(timezone=True), nullable=True)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    exam_sessions = relationship("ExamSession", back_populates="student")
    voice_profiles = relationship("VoiceProfile", back_populates="student")
    
    def __repr__(self):
        return f"<Student(id='{self.student_id}', name='{self.full_name}')>"

class Exam(Base):
    """Exam definition model."""
    __tablename__ = "exams"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    exam_code = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Exam timing
    scheduled_start = Column(DateTime(timezone=True), nullable=False)
    scheduled_end = Column(DateTime(timezone=True), nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    
    # Exam settings
    max_attempts = Column(Integer, default=1, nullable=False)
    requires_voice_auth = Column(Boolean, default=True, nullable=False)
    lockdown_level = Column(String(20), default="strict", nullable=False)  # strict, moderate, basic
    
    # Security settings
    allow_copy_paste = Column(Boolean, default=False, nullable=False)
    allow_right_click = Column(Boolean, default=False, nullable=False)
    monitor_audio = Column(Boolean, default=True, nullable=False)
    monitor_video = Column(Boolean, default=False, nullable=False)
    
    # Status
    status = Column(String(20), default=ExamStatus.SCHEDULED, nullable=False)
    
    # Configuration
    configuration = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    exam_sessions = relationship("ExamSession", back_populates="exam")
    
    def __repr__(self):
        return f"<Exam(code='{self.exam_code}', title='{self.title}')>"

class ExamSession(Base):
    """Individual exam session model."""
    __tablename__ = "exam_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_token = Column(String(128), unique=True, nullable=False, index=True)
    
    # Foreign keys
    student_id = Column(String(36), ForeignKey("students.id"), nullable=False, index=True)
    exam_id = Column(String(36), ForeignKey("exams.id"), nullable=False, index=True)
    
    # Session details
    attempt_number = Column(Integer, default=1, nullable=False)
    status = Column(String(20), default=SessionStatus.PENDING, nullable=False, index=True)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    last_activity = Column(DateTime(timezone=True), nullable=True)
    time_remaining = Column(Integer, nullable=True)  # seconds
    
    # Authentication
    voice_verified = Column(Boolean, default=False, nullable=False)
    auth_timestamp = Column(DateTime(timezone=True), nullable=True)
    auth_confidence = Column(Float, nullable=True)
    
    # Monitoring
    monitoring_active = Column(Boolean, default=False, nullable=False)
    security_violations = Column(Integer, default=0, nullable=False)
    warning_count = Column(Integer, default=0, nullable=False)
    
    # Environment
    browser_info = Column(JSON, nullable=True)
    system_info = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    # Results
    score = Column(Float, nullable=True)
    max_score = Column(Float, nullable=True)
    submission_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    student = relationship("Student", back_populates="exam_sessions")
    exam = relationship("Exam", back_populates="exam_sessions")
    monitoring_events = relationship("MonitoringEvent", back_populates="exam_session")
    security_alerts = relationship("SecurityAlert", back_populates="exam_session")
    
    def __repr__(self):
        return f"<ExamSession(id='{self.id}', status='{self.status}')>"

class VoiceProfile(Base):
    """Voice profile model for student authentication."""
    __tablename__ = "voice_profiles"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String(36), ForeignKey("students.id"), nullable=False, index=True)
    
    # Profile details
    profile_hash = Column(String(64), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)
    snr_ratio = Column(Float, nullable=True)  # Signal-to-noise ratio
    
    # Usage statistics
    enrollment_date = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    student = relationship("Student", back_populates="voice_profiles")
    
    def __repr__(self):
        return f"<VoiceProfile(student_id='{self.student_id}')>"

class MonitoringEvent(Base):
    """Real-time monitoring events during exams."""
    __tablename__ = "monitoring_events"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("exam_sessions.id"), nullable=False, index=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # voice_check, app_switch, etc.
    event_category = Column(String(30), nullable=False, index=True)  # security, system, user
    severity = Column(String(20), default=AlertSeverity.LOW, nullable=False, index=True)
    
    # Event data
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    event_data = Column(JSON, nullable=True)
    
    # Analysis results
    confidence_score = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)
    automated_action = Column(String(50), nullable=True)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    duration = Column(Float, nullable=True)  # Event duration in seconds
    
    # Status
    acknowledged = Column(Boolean, default=False, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    exam_session = relationship("ExamSession", back_populates="monitoring_events")
    
    def __repr__(self):
        return f"<MonitoringEvent(type='{self.event_type}', severity='{self.severity}')>"

class SecurityAlert(Base):
    """Security alerts for potential cheating or violations."""
    __tablename__ = "security_alerts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("exam_sessions.id"), nullable=False, index=True)
    
    # Alert details
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Detection details
    detection_method = Column(String(50), nullable=True)  # voice_analysis, behavior_pattern, etc.
    confidence_score = Column(Float, nullable=True)
    evidence_data = Column(JSON, nullable=True)
    
    # Response
    automated_action = Column(String(50), nullable=True)  # warning, suspend, terminate
    manual_review_required = Column(Boolean, default=False, nullable=False)
    
    # Status
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    
    resolved = Column(Boolean, default=False, nullable=False)
    resolution_action = Column(String(100), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    exam_session = relationship("ExamSession", back_populates="security_alerts")
    
    def __repr__(self):
        return f"<SecurityAlert(type='{self.alert_type}', severity='{self.severity}')>"

class AuditLog(Base):
    """Comprehensive audit log for compliance and forensics."""
    __tablename__ = "audit_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Context
    session_id = Column(String(36), nullable=True, index=True)
    student_id = Column(String(36), nullable=True, index=True)
    exam_id = Column(String(36), nullable=True, index=True)
    
    # Action details
    action_type = Column(String(50), nullable=False, index=True)
    action_category = Column(String(30), nullable=False, index=True)
    action_description = Column(Text, nullable=False)
    
    # Actor information
    actor_type = Column(String(20), nullable=False)  # student, system, admin
    actor_id = Column(String(100), nullable=True)
    
    # Technical details
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    
    # Result
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<AuditLog(action='{self.action_type}', success={self.success})>"

class SystemMetrics(Base):
    """System performance and usage metrics."""
    __tablename__ = "system_metrics"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Metric details
    metric_type = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    
    # Context
    context = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.metric_value})>"

# Create composite indexes for better performance
def create_exam_indexes(engine):
    """Create specialized indexes for exam system performance."""
    with engine.connect() as conn:
        # Exam session performance indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_exam_sessions_student_status 
            ON exam_sessions(student_id, status, started_at DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_exam_sessions_exam_active 
            ON exam_sessions(exam_id, status) WHERE status IN ('in_progress', 'authenticating')
        """))
        
        # Monitoring events performance indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_monitoring_events_session_time 
            ON monitoring_events(session_id, timestamp DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_monitoring_events_severity_time 
            ON monitoring_events(severity, timestamp DESC) WHERE severity IN ('high', 'critical')
        """))
        
        # Security alerts performance indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_security_alerts_unresolved 
            ON security_alerts(session_id, timestamp DESC) WHERE resolved = false
        """))
        
        # Audit logs performance indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_session_time 
            ON audit_logs(session_id, timestamp DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_student_action 
            ON audit_logs(student_id, action_type, timestamp DESC)
        """))
        
        conn.commit()
