"""
Database Connection Manager for Voice Authentication System

This module provides database connection management, session handling,
and database initialization for the voice authentication system.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import logging
import time
from typing import Generator, Optional
from pathlib import Path

from app.config import settings
from app.database.models import Base, create_indexes

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database connection manager with support for multiple database types.
    
    Features:
    - Automatic connection pooling
    - Health checks and reconnection
    - Performance monitoring
    - Support for SQLite, PostgreSQL, MySQL, SQL Server
    """
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._is_initialized = False
        self.setup_database()
    
    def setup_database(self):
        """Setup database connection and create tables."""
        try:
            logger.info(f"Setting up database connection: {self._get_safe_url()}")
            
            # Create engine based on database type
            self.engine = self._create_engine()
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables if they don't exist
            self._create_tables()
            
            # Create additional indexes
            self._create_indexes()
            
            # Test connection
            self._test_connection()
            
            self._is_initialized = True
            logger.info("Database setup completed successfully")
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            raise
    
    def _create_engine(self):
        """Create SQLAlchemy engine based on database URL."""
        database_url = settings.DATABASE_URL
        
        if database_url.startswith('sqlite'):
            return self._create_sqlite_engine(database_url)
        elif database_url.startswith('postgresql'):
            return self._create_postgresql_engine(database_url)
        elif database_url.startswith('mysql'):
            return self._create_mysql_engine(database_url)
        elif database_url.startswith('mssql'):
            return self._create_mssql_engine(database_url)
        else:
            # Generic engine for other databases
            return create_engine(
                database_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False  # Set to True for SQL debugging
            )
    
    def _create_sqlite_engine(self, database_url: str):
        """Create SQLite engine with optimized settings."""
        # Ensure data directory exists
        if '///' in database_url:
            db_path = database_url.split('///')[-1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        engine = create_engine(
            database_url,
            connect_args={
                "check_same_thread": False,
                "timeout": 20,
            },
            poolclass=StaticPool,
            echo=False
        )
        
        # Enable WAL mode for better concurrency
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()
        
        return engine
    
    def _create_postgresql_engine(self, database_url: str):
        """Create PostgreSQL engine with optimized settings."""
        return create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=QueuePool,
            echo=False,
            connect_args={
                "connect_timeout": 10,
                "application_name": "voice_auth_system"
            }
        )
    
    def _create_mysql_engine(self, database_url: str):
        """Create MySQL engine with optimized settings."""
        return create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=QueuePool,
            echo=False,
            connect_args={
                "connect_timeout": 10,
                "charset": "utf8mb4"
            }
        )
    
    def _create_mssql_engine(self, database_url: str):
        """Create SQL Server engine with optimized settings."""
        return create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=QueuePool,
            echo=False,
            connect_args={
                "timeout": 10
            }
        )
    
    def _create_tables(self):
        """Create database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create additional database indexes."""
        try:
            create_indexes(self.engine)
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {str(e)}")
            # Don't raise here as indexes are optional optimizations
    
    def _test_connection(self):
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                if result:
                    logger.info("Database connection test successful")
                else:
                    raise Exception("Connection test returned no result")
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise
    
    def _get_safe_url(self) -> str:
        """Get database URL with password masked for logging."""
        url = settings.DATABASE_URL
        if '@' in url:
            # Mask password in URL for logging
            parts = url.split('@')
            if len(parts) == 2:
                user_pass = parts[0].split('//')[-1]
                if ':' in user_pass:
                    user = user_pass.split(':')[0]
                    return url.replace(user_pass, f"{user}:***")
        return url
    
    def get_session(self) -> Session:
        """
        Get database session.
        
        Returns:
            SQLAlchemy session instance
        """
        if not self._is_initialized:
            raise RuntimeError("Database manager not initialized")
        
        return self.SessionLocal()
    
    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        """
        Get database session as context manager.
        
        Yields:
            SQLAlchemy session instance
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def health_check(self) -> dict:
        """
        Perform database health check.
        
        Returns:
            Dictionary with health check results
        """
        try:
            start_time = time.time()
            
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "database_type": self.engine.dialect.name,
                "pool_size": getattr(self.engine.pool, 'size', None),
                "checked_out_connections": getattr(self.engine.pool, 'checkedout', None),
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_type": self.engine.dialect.name if self.engine else "unknown",
            }
    
    def get_connection_info(self) -> dict:
        """
        Get database connection information.
        
        Returns:
            Dictionary with connection information
        """
        if not self.engine:
            return {"status": "not_initialized"}
        
        return {
            "database_type": self.engine.dialect.name,
            "database_version": getattr(self.engine.dialect, 'server_version_info', None),
            "pool_size": getattr(self.engine.pool, 'size', None),
            "max_overflow": getattr(self.engine.pool, '_max_overflow', None),
            "pool_timeout": getattr(self.engine.pool, '_timeout', None),
            "pool_recycle": getattr(self.engine.pool, '_recycle', None),
        }
    
    def close(self):
        """Close database connection and cleanup resources."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()

# Global database manager instance
db_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """
    Get global database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session for FastAPI.
    
    Yields:
        SQLAlchemy session instance
    """
    manager = get_database_manager()
    session = manager.get_session()
    try:
        yield session
    finally:
        session.close()

def get_db_session() -> Session:
    """
    Get database session for direct use.
    
    Returns:
        SQLAlchemy session instance
    """
    manager = get_database_manager()
    return manager.get_session()

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Get database session as context manager.
    
    Yields:
        SQLAlchemy session instance
    """
    manager = get_database_manager()
    with manager.get_session_context() as session:
        yield session

def init_database():
    """Initialize database connection."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def close_database():
    """Close database connections."""
    global db_manager
    if db_manager:
        db_manager.close()
        db_manager = None
