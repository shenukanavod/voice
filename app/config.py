import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    DATABASE_URL: str = "sqlite:///./data/voice_auth.db"

    # MongoDB Atlas settings (cloud database - works from any device)
    MONGODB_URL: str = "mongodb+srv://admin:voice123@cluster0.gjap7.mongodb.net/voice_auth_db?retryWrites=true&w=majority&appName=Cluster0&tlsAllowInvalidCertificates=true"
    DATABASE_NAME: str = "voice_auth_db"
    USE_MONGODB: bool = True  # Use MongoDB for voice profile storage

    # Audio processing settings
    SAMPLE_RATE: int = 16000
    AUDIO_DURATION: float = 3.0  # seconds
    N_MFCC: int = 13
    N_MELS: int = 128
    HOP_LENGTH: int = 512
    N_FFT: int = 2048

    # Model settings
    MODEL_PATH: str = "data/models"
    VOICE_PROFILES_PATH: str = "data/voice_profiles"

    # Authentication thresholds
    # CRITICAL: Threshold determines similarity required for authentication
    # 0.80 = 80% match required (balanced security)
    # 0.85 = 85% match (good security)
    # 0.90 = 90% match (high security)
    # 0.95 = 95% match (very high security)
    # ⚠️  IMPORTANT: With model accuracy 83.86%, use appropriate threshold to prevent cross-user authentication
    VERIFICATION_THRESHOLD: float = 0.80  # Balanced threshold for 83.86% accuracy model
    SPOOFING_THRESHOLD: float = 0.5
    CONTINUOUS_MONITORING_INTERVAL: float = 2.0  # seconds

    # Security settings
    ENCRYPTION_KEY_PATH: str = "data/encryption.key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Performance settings
    MAX_ENROLLMENT_SAMPLES: int = 5
    BATCH_SIZE: int = 32

    class Config:
        env_file = ".env"


# Create directories if they don't exist
def create_directories():
    directories = ["data", "data/models", "data/voice_profiles", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


settings = Settings()
create_directories()
