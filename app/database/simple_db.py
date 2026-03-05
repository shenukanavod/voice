"""
Simple Local Database Manager for Voice Authentication

This provides a working database solution while MongoDB Atlas 
connection is being configured.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleDBManager:
    """Simple SQLite database manager for voice authentication."""
    
    def __init__(self, db_path: str = "data/voice_auth.db"):
        self.db_path = db_path
        self.ensure_database_exists()
        self.create_tables()
    
    def ensure_database_exists(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def create_tables(self):
        """Create necessary tables."""
        with self.get_connection() as conn:
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    full_name TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    is_enrolled BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)
            
            # Voice profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    profile_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_verified TIMESTAMP,
                    verification_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    metadata TEXT
                )
            """)
            
            # Authentication logs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS authentication_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    authentication_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    similarity_score REAL,
                    threshold_used REAL,
                    processing_time REAL,
                    error_message TEXT,
                    ip_address TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("✅ Database tables created/verified")
    
    def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create a new user."""
        import uuid
        
        user_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO users (id, username, email, full_name, is_active, is_enrolled)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                user_data.get('username'),
                user_data.get('email'),
                user_data.get('full_name'),
                user_data.get('is_active', True),
                user_data.get('is_enrolled', False)
            ))
            conn.commit()
        
        logger.info(f"✅ User created: {user_data.get('username')}")
        return user_id
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def create_voice_profile(self, profile_data: Dict[str, Any]) -> str:
        """Create voice profile."""
        import uuid
        
        profile_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO voice_profiles 
                (id, user_id, profile_hash, file_path, file_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                profile_id,
                profile_data.get('user_id'),
                profile_data.get('profile_hash'),
                profile_data.get('file_path'),
                profile_data.get('file_size'),
                json.dumps(profile_data.get('metadata', {}))
            ))
            conn.commit()
        
        logger.info(f"✅ Voice profile created for user: {profile_data.get('user_id')}")
        return profile_id
    
    def get_voice_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get voice profile for user."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM voice_profiles 
                WHERE user_id = ? AND is_active = 1
                ORDER BY enrollment_date DESC
                LIMIT 1
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                profile = dict(row)
                if profile.get('metadata'):
                    profile['metadata'] = json.loads(profile['metadata'])
                return profile
            return None
    
    def log_authentication(self, log_data: Dict[str, Any]) -> str:
        """Log authentication attempt."""
        import uuid
        
        log_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO authentication_logs 
                (id, user_id, authentication_type, success, similarity_score, 
                 threshold_used, processing_time, error_message, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_id,
                log_data.get('user_id'),
                log_data.get('authentication_type'),
                log_data.get('success'),
                log_data.get('similarity_score'),
                log_data.get('threshold_used'),
                log_data.get('processing_time'),
                log_data.get('error_message'),
                log_data.get('ip_address')
            ))
            conn.commit()
        
        return log_id
    
    def get_enrolled_users(self) -> List[Dict[str, Any]]:
        """Get all enrolled users."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT u.*, 
                       vp.enrollment_date,
                       vp.last_verified,
                       vp.verification_count,
                       vp.success_count
                FROM users u
                LEFT JOIN voice_profiles vp ON u.id = vp.user_id AND vp.is_active = 1
                WHERE u.is_enrolled = 1 AND u.is_active = 1
                ORDER BY vp.enrollment_date DESC
            """)
            
            users = []
            for row in cursor.fetchall():
                user = dict(row)
                # Calculate success rate
                if user['verification_count'] and user['verification_count'] > 0:
                    user['success_rate'] = user['success_count'] / user['verification_count']
                else:
                    user['success_rate'] = 0.0
                users.append(user)
            
            return users
    
    def update_user_enrollment(self, username: str, is_enrolled: bool = True):
        """Update user enrollment status."""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE users 
                SET is_enrolled = ?, last_login = CURRENT_TIMESTAMP
                WHERE username = ?
            """, (is_enrolled, username))
            conn.commit()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            stats = {}
            
            # Count users
            cursor = conn.execute("SELECT COUNT(*) as count FROM users")
            stats['total_users'] = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM users WHERE is_enrolled = 1")
            stats['enrolled_users'] = cursor.fetchone()['count']
            
            # Count voice profiles
            cursor = conn.execute("SELECT COUNT(*) as count FROM voice_profiles WHERE is_active = 1")
            stats['voice_profiles'] = cursor.fetchone()['count']
            
            # Count authentication logs
            cursor = conn.execute("SELECT COUNT(*) as count FROM authentication_logs")
            stats['authentication_attempts'] = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM authentication_logs WHERE success = 1")
            stats['successful_authentications'] = cursor.fetchone()['count']
            
            return stats

# Global database instance
_db_manager = None

def get_db_manager() -> SimpleDBManager:
    """Get or create database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = SimpleDBManager()
    
    return _db_manager

def initialize_sample_data():
    """Initialize with some sample data for testing."""
    db = get_db_manager()
    
    # Check if sample user already exists
    existing_user = db.get_user("demo_user")
    if not existing_user:
        # Create sample user
        user_data = {
            "username": "demo_user",
            "email": "demo@example.com",
            "full_name": "Demo User",
            "is_active": True,
            "is_enrolled": False
        }
        
        user_id = db.create_user(user_data)
        logger.info(f"✅ Sample user created: {user_id}")
        
        return user_id
    else:
        logger.info("✅ Sample user already exists")
        return existing_user['id']