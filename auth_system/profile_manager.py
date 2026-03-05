"""
Voice Profile Manager
Handles storage and retrieval of user voice profiles
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import pickle


class VoiceProfileManager:
    """Manages user voice profiles and embeddings."""
    
    def __init__(self, profiles_dir: str = "voice_profiles"):
        """
        Initialize profile manager.
        
        Args:
            profiles_dir: Directory to store voice profiles
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Profile manager initialized")
        print(f"   Profiles directory: {self.profiles_dir}")
    
    def save_profile(self, 
                    user_id: str, 
                    embedding: np.ndarray,
                    metadata: Optional[Dict] = None) -> bool:
        """
        Save user voice profile.
        
        Args:
            user_id: Unique user identifier
            embedding: Voice embedding vector
            metadata: Optional metadata (e.g., enrollment date, audio quality)
            
        Returns:
            Success status
        """
        try:
            # Create user directory
            user_dir = self.profiles_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Save embedding
            embedding_path = user_dir / "embedding.npy"
            np.save(embedding_path, embedding)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata['user_id'] = user_id
            metadata['enrollment_date'] = datetime.now().isoformat()
            metadata['embedding_shape'] = embedding.shape
            
            metadata_path = user_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Profile saved for user: {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving profile: {e}")
            return False
    
    def load_profile(self, user_id: str) -> Optional[np.ndarray]:
        """
        Load user voice profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            Voice embedding or None if not found
        """
        embedding_path = self.profiles_dir / user_id / "embedding.npy"
        
        if not embedding_path.exists():
            return None
        
        try:
            embedding = np.load(embedding_path)
            return embedding
        except Exception as e:
            print(f"❌ Error loading profile: {e}")
            return None
    
    def get_metadata(self, user_id: str) -> Optional[Dict]:
        """
        Get user metadata.
        
        Args:
            user_id: User identifier
            
        Returns:
            Metadata dictionary or None
        """
        metadata_path = self.profiles_dir / user_id / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None
    
    def delete_profile(self, user_id: str) -> bool:
        """
        Delete user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            Success status
        """
        user_dir = self.profiles_dir / user_id
        
        if not user_dir.exists():
            print(f"⚠️  Profile not found: {user_id}")
            return False
        
        try:
            # Delete all files in user directory
            for file in user_dir.iterdir():
                file.unlink()
            
            # Delete directory
            user_dir.rmdir()
            
            print(f"✅ Profile deleted: {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting profile: {e}")
            return False
    
    def list_users(self) -> List[str]:
        """
        List all enrolled users.
        
        Returns:
            List of user IDs
        """
        users = []
        
        for user_dir in self.profiles_dir.iterdir():
            if user_dir.is_dir():
                users.append(user_dir.name)
        
        return sorted(users)
    
    def user_exists(self, user_id: str) -> bool:
        """
        Check if user profile exists.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if profile exists
        """
        embedding_path = self.profiles_dir / user_id / "embedding.npy"
        return embedding_path.exists()
    
    def get_profile_count(self) -> int:
        """
        Get total number of enrolled profiles.
        
        Returns:
            Number of profiles
        """
        return len(self.list_users())
