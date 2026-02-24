"""
Voice Authentication System
Main system integrating all components for enrollment and verification
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import cosine

from .model_utils import EmbeddingExtractor
from .audio_processor import AudioProcessor
from .profile_manager import VoiceProfileManager


class VoiceAuthSystem:
    """Complete voice authentication system."""
    
    def __init__(self,
                 model_path: str = "models/speaker_embedding_model.pth",
                 profiles_dir: str = "voice_profiles",
                 sample_rate: int = 16000,
                 duration: float = 3.0,
                 threshold: float = 0.75):
        """
        Initialize voice authentication system.
    
        """
        print("=" * 70)
        print("VOICE AUTHENTICATION SYSTEM")
        print("=" * 70)
        
        # Initialize components
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            duration=duration
        )
        
        self.embedding_extractor = EmbeddingExtractor(model_path)
        
        self.profile_manager = VoiceProfileManager(profiles_dir)
        
        self.threshold = threshold
        
        print(f"✅ System initialized")
        print(f"   Verification threshold: {threshold:.2f} ({threshold*100:.0f}% similarity)")
        print("=" * 70)
        print()
    
    def enroll_user(self, user_id: str, audio_source: str = "microphone") -> Tuple[bool, str]:
        """
        Enroll a new user.
        
      
        """
        print(f"\n{'='*70}")
        print(f"ENROLLMENT - User: {user_id}")
        print(f"{'='*70}\n")
        
        try:
            # Check if user already exists
            if self.profile_manager.user_exists(user_id):
                overwrite = input(f"User {user_id} already exists. Overwrite? (yes/no): ")
                if overwrite.lower() != 'yes':
                    return False, "Enrollment cancelled"
            
            # Record or load audio
            if audio_source == "microphone":
                audio = self.audio_processor.record_audio()
            else:
                print(f" Loading audio from: {audio_source}")
                audio = self.audio_processor.load_audio(audio_source)
            
            # Validate audio
            is_valid, validation_msg = self.audio_processor.validate_audio(audio)
            if not is_valid:
                return False, f"Audio validation failed: {validation_msg}"
            
            print(f" {validation_msg}")
            
            # Extract features
            print(" Extracting voice features...")
            mel_spec = self.audio_processor.preprocess_audio(audio)
            
            # Generate embedding
            print(" Generating voice embedding...")
            embedding = self.embedding_extractor.extract_embedding(mel_spec)
            
            # Save profile
            metadata = {
                'audio_source': audio_source,
                'sample_rate': self.audio_processor.sample_rate,
                'duration': self.audio_processor.duration,
                'embedding_dim': len(embedding)
            }
            
            success = self.profile_manager.save_profile(user_id, embedding, metadata)
            
            if success:
                return True, f"User {user_id} enrolled successfully!"
            else:
                return False, "Failed to save profile"
                
        except Exception as e:
            return False, f"Enrollment error: {str(e)}"
    
    def verify_user(self, user_id: str, audio_source: str = "microphone") -> Tuple[bool, float, str]:
        """
        Verify a user's identity.
       
        """
        print(f"\n{'='*70}")
        print(f"VERIFICATION - User: {user_id}")
        print(f"{'='*70}\n")
        
        try:
            # Check if user exists
            if not self.profile_manager.user_exists(user_id):
                return False, 0.0, f"User {user_id} not enrolled"
            
            # Load enrolled profile
            enrolled_embedding = self.profile_manager.load_profile(user_id)
            if enrolled_embedding is None:
                return False, 0.0, "Failed to load enrolled profile"
            
            print(f" Loaded profile for {user_id}")
            
            # Record or load audio
            if audio_source == "microphone":
                audio = self.audio_processor.record_audio()
            else:
                print(f" Loading audio from: {audio_source}")
                audio = self.audio_processor.load_audio(audio_source)
            
            # Validate audio
            is_valid, validation_msg = self.audio_processor.validate_audio(audio)
            if not is_valid:
                return False, 0.0, f"Audio validation failed: {validation_msg}"
            
            # Extract features
            print(" Extracting voice features...")
            mel_spec = self.audio_processor.preprocess_audio(audio)
            
            # Generate embedding
            print(" Generating voice embedding...")
            verification_embedding = self.embedding_extractor.extract_embedding(mel_spec)
            
            # Calculate similarity
            similarity = self._calculate_similarity(enrolled_embedding, verification_embedding)
            
            print(f"\n Similarity score: {similarity:.4f} ({similarity*100:.2f}%)")
            print(f"   Threshold: {self.threshold:.4f} ({self.threshold*100:.0f}%)")
            
            # Verify
            is_verified = similarity >= self.threshold
            
            if is_verified:
                message = f" VERIFIED - {user_id} authenticated successfully"
            else:
                message = f" NOT VERIFIED - Similarity below threshold"
            
            return is_verified, similarity, message
            
        except Exception as e:
            return False, 0.0, f"Verification error: {str(e)}"
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings.
       
        """
        # Cosine similarity: 1 - cosine_distance
        similarity = 1 - cosine(emb1, emb2)
        return float(similarity)
    
    def list_enrolled_users(self) -> None:
        """Display all enrolled users."""
        users = self.profile_manager.list_users()
        
        print(f"\n{'='*70}")
        print(f"ENROLLED USERS ({len(users)})")
        print(f"{'='*70}\n")
        
        if not users:
            print("No users enrolled yet.")
        else:
            for i, user_id in enumerate(users, 1):
                metadata = self.profile_manager.get_metadata(user_id)
                enrollment_date = metadata.get('enrollment_date', 'Unknown') if metadata else 'Unknown'
                print(f"{i}. {user_id}")
                print(f"   Enrolled: {enrollment_date[:10] if enrollment_date != 'Unknown' else 'Unknown'}")
        
        print()
    
    def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """
        Delete a user profile.
       
        """
        if not self.profile_manager.user_exists(user_id):
            return False, f"User {user_id} not found"
        
        success = self.profile_manager.delete_profile(user_id)
        
        if success:
            return True, f"User {user_id} deleted successfully"
        else:
            return False, "Failed to delete user"
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Update verification threshold.
        
        """
        if not 0 <= new_threshold <= 1:
            print("  Threshold must be between 0 and 1")
            return
        
        self.threshold = new_threshold
        print(f" Threshold updated to {new_threshold:.2f} ({new_threshold*100:.0f}%)")
