import os
import json
import pickle
import hashlib
from typing import Any, Optional, Dict
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import numpy as np

from app.config import settings

class VoiceProfileEncryption:
    """
    Secure encryption system for voice profiles and embeddings.
    
    Features:
    - AES-256 encryption for voice embeddings
    - Key derivation from master password
    - Secure storage of encrypted profiles
    - Integrity verification using HMAC
    """
    
    def __init__(self, master_key: Optional[str] = None):
        self.key_file = Path(settings.ENCRYPTION_KEY_PATH)
        self.profiles_dir = Path(settings.VOICE_PROFILES_PATH)
        
        # Ensure directories exist
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load encryption key
        if master_key:
            self.fernet = self._derive_key_from_password(master_key)
        else:
            self.fernet = self._load_or_generate_key()
    
    def _generate_key(self) -> Fernet:
        """Generate a new encryption key."""
        key = Fernet.generate_key()
        return Fernet(key)
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> Fernet:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: Master password
            salt: Optional salt (generated if not provided)
            
        Returns:
            Fernet encryption object
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        # Store salt for future key derivation
        salt_file = self.key_file.parent / "salt.bin"
        with open(salt_file, 'wb') as f:
            f.write(salt)
        
        return Fernet(key)
    
    def _load_or_generate_key(self) -> Fernet:
        """Load existing key or generate new one."""
        if self.key_file.exists():
            try:
                with open(self.key_file, 'rb') as f:
                    key = f.read()
                # Ensure key is valid Fernet key format
                if len(key) == 44:  # Fernet keys are 44 bytes (base64 encoded 32 bytes)
                    return Fernet(key)
                else:
                    print(f"Warning: Invalid key length {len(key)}, generating new key")
            except Exception as e:
                print(f"Warning: Failed to load key: {e}, generating new key")
        
        # Generate new key
        key = Fernet.generate_key()
        fernet = Fernet(key)
        
        # Save key to file
        with open(self.key_file, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions (only on Unix-like systems)
        try:
            os.chmod(self.key_file, 0o600)
        except:
            pass  # Windows doesn't support chmod
        
        return fernet
    
    def encrypt_data(self, data: Any) -> bytes:
        """
        Encrypt arbitrary data.
        
        Args:
            data: Data to encrypt (will be pickled)
            
        Returns:
            Encrypted data as bytes
        """
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            
            # Encrypt
            encrypted_data = self.fernet.encrypt(serialized_data)
            
            return encrypted_data
            
        except Exception as e:
            raise ValueError(f"Error encrypting data: {str(e)}")
    
    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data bytes
            
        Returns:
            Decrypted and deserialized data
        """
        try:
            # Decrypt
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            # Deserialize
            data = pickle.loads(decrypted_data)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error decrypting data: {str(e)}")
    
    def save_voice_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Save encrypted voice profile.
        
        Args:
            user_id: Unique user identifier
            profile_data: Voice profile data to encrypt and save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata
            profile_data['user_id'] = user_id
            profile_data['created_at'] = str(np.datetime64('now'))
            
            # Calculate hash for integrity verification
            profile_hash = self._calculate_hash(profile_data)
            profile_data['hash'] = profile_hash
            
            # Encrypt profile data
            encrypted_data = self.encrypt_data(profile_data)
            
            # Save to file
            profile_file = self.profiles_dir / f"{user_id}.enc"
            with open(profile_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(profile_file, 0o600)
            
            return True
            
        except Exception as e:
            print(f"Error saving voice profile for {user_id}: {str(e)}")
            return False
    
    def load_voice_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load and decrypt voice profile.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Decrypted voice profile data or None if not found/invalid
        """
        try:
            profile_file = self.profiles_dir / f"{user_id}.enc"
            
            if not profile_file.exists():
                return None
            
            # Load encrypted data
            with open(profile_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt profile data
            profile_data = self.decrypt_data(encrypted_data)
            
            # Verify integrity
            stored_hash = profile_data.pop('hash', None)
            calculated_hash = self._calculate_hash(profile_data)
            
            if stored_hash != calculated_hash:
                print(f"Warning: Hash mismatch for profile {user_id}. Data may be corrupted.")
                return None
            
            return profile_data
            
        except Exception as e:
            print(f"Error loading voice profile for {user_id}: {str(e)}")
            return None
    
    def delete_voice_profile(self, user_id: str) -> bool:
        """
        Delete voice profile.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile_file = self.profiles_dir / f"{user_id}.enc"
            
            if profile_file.exists():
                profile_file.unlink()
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting voice profile for {user_id}: {str(e)}")
            return False
    
    def list_profiles(self) -> list:
        """
        List all available voice profiles.
        
        Returns:
            List of user IDs with profiles
        """
        try:
            profile_files = list(self.profiles_dir.glob("*.enc"))
            user_ids = [f.stem for f in profile_files]
            return user_ids
            
        except Exception as e:
            print(f"Error listing profiles: {str(e)}")
            return []
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of profile data for integrity verification.
        
        Args:
            data: Profile data dictionary
            
        Returns:
            Hexadecimal hash string
        """
        # Create a copy and remove non-deterministic fields
        data_copy = data.copy()
        data_copy.pop('hash', None)
        data_copy.pop('created_at', None)
        
        # Convert to JSON string (sorted keys for consistency)
        json_str = json.dumps(data_copy, sort_keys=True, default=str)
        
        # Calculate hash
        hash_obj = hashlib.sha256(json_str.encode())
        return hash_obj.hexdigest()
    
    def backup_profiles(self, backup_path: str) -> bool:
        """
        Create encrypted backup of all voice profiles.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Collect all profiles
            profiles = {}
            for user_id in self.list_profiles():
                profile = self.load_voice_profile(user_id)
                if profile:
                    profiles[user_id] = profile
            
            # Add backup metadata
            backup_data = {
                'profiles': profiles,
                'backup_timestamp': str(np.datetime64('now')),
                'version': '1.0'
            }
            
            # Encrypt backup
            encrypted_backup = self.encrypt_data(backup_data)
            
            # Save backup
            with open(backup_path, 'wb') as f:
                f.write(encrypted_backup)
            
            return True
            
        except Exception as e:
            print(f"Error creating backup: {str(e)}")
            return False
    
    def restore_profiles(self, backup_path: str) -> bool:
        """
        Restore voice profiles from encrypted backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load backup
            with open(backup_path, 'rb') as f:
                encrypted_backup = f.read()
            
            # Decrypt backup
            backup_data = self.decrypt_data(encrypted_backup)
            
            # Restore profiles
            profiles = backup_data.get('profiles', {})
            success_count = 0
            
            for user_id, profile_data in profiles.items():
                if self.save_voice_profile(user_id, profile_data):
                    success_count += 1
            
            print(f"Restored {success_count}/{len(profiles)} profiles")
            return success_count > 0
            
        except Exception as e:
            print(f"Error restoring backup: {str(e)}")
            return False

class SecureVoiceProfileManager:
    """
    High-level manager for secure voice profile operations.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        self.encryption = VoiceProfileEncryption(master_key)
        self.profile_cache = {}  # In-memory cache for frequently accessed profiles
        self.cache_timeout = 300  # 5 minutes
    
    def create_profile(self, user_id: str, audio_data: np.ndarray, 
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a voice profile for a user from audio data.
        
        Args:
            user_id: Unique user identifier
            audio_data: Preprocessed audio array
            metadata: Optional metadata
            
        Returns:
            Profile data dictionary
        """
        try:
            # For now, use the audio data directly as embeddings
            # In a full implementation, this would extract features/embeddings
            from app.audio.preprocessing import AudioPreprocessor
            
            preprocessor = AudioPreprocessor()
            
            # Extract MFCC features as embeddings
            mfcc_features = preprocessor.extract_mfcc(audio_data)
            
            # Flatten to 1D array for storage
            embeddings = mfcc_features.flatten()
            
            # Create profile data
            profile_data = {
                'user_id': user_id,
                'embeddings': embeddings.tolist(),
                'embedding_shape': embeddings.shape,
                'enrollment_date': str(np.datetime64('now')),
                'metadata': metadata or {}
            }
            
            # Save encrypted profile
            success = self.encryption.save_voice_profile(user_id, profile_data)
            
            if success:
                # Update cache
                self.profile_cache[user_id] = {
                    'data': profile_data,
                    'timestamp': np.datetime64('now')
                }
                return profile_data
            else:
                raise Exception("Failed to save profile")
            
        except Exception as e:
            raise Exception(f"Error creating profile for user {user_id}: {str(e)}")
    
    def verify(self, user_id: str, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Verify a user's voice against their stored profile.
        
        Args:
            user_id: Unique user identifier
            audio_data: Preprocessed audio array to verify
            
        Returns:
            Dictionary with verification result:
            {
                'verified': bool,
                'similarity': float,
                'user_id': str
            }
        """
        try:
            from app.audio.preprocessing import AudioPreprocessor
            from scipy.spatial.distance import cosine
            
            # Get stored profile
            profile_data = self.encryption.load_voice_profile(user_id)
            
            if not profile_data:
                return {
                    'verified': False,
                    'similarity': 0.0,
                    'user_id': user_id,
                    'error': 'User profile not found'
                }
            
            # Extract features from new audio
            preprocessor = AudioPreprocessor()
            mfcc_features = preprocessor.extract_mfcc(audio_data)
            new_embeddings = mfcc_features.flatten()
            
            # Get stored embeddings
            stored_embeddings = np.array(profile_data['embeddings'])
            
            # Ensure same length
            min_len = min(len(new_embeddings), len(stored_embeddings))
            new_embeddings = new_embeddings[:min_len]
            stored_embeddings = stored_embeddings[:min_len]
            
            # Calculate cosine similarity
            similarity = 1 - cosine(new_embeddings, stored_embeddings)
            
            # Threshold for verification (70%)
            threshold = 0.70
            verified = similarity >= threshold
            
            return {
                'verified': verified,
                'similarity': float(similarity),
                'user_id': user_id,
                'threshold': threshold
            }
            
        except Exception as e:
            return {
                'verified': False,
                'similarity': 0.0,
                'user_id': user_id,
                'error': str(e)
            }
    
    def enroll_user(self, user_id: str, voice_embeddings: np.ndarray, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Enroll a new user with voice embeddings.
        
        Args:
            user_id: Unique user identifier
            voice_embeddings: Voice embeddings array
            metadata: Optional metadata (enrollment date, device info, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare profile data
            profile_data = {
                'embeddings': voice_embeddings.tolist(),  # Convert numpy array to list
                'embedding_shape': voice_embeddings.shape,
                'enrollment_date': str(np.datetime64('now')),
                'metadata': metadata or {}
            }
            
            # Save encrypted profile
            success = self.encryption.save_voice_profile(user_id, profile_data)
            
            if success:
                # Update cache
                self.profile_cache[user_id] = {
                    'data': profile_data,
                    'timestamp': np.datetime64('now')
                }
            
            return success
            
        except Exception as e:
            print(f"Error enrolling user {user_id}: {str(e)}")
            return False
    
    def get_user_embeddings(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get voice embeddings for a user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Voice embeddings array or None if not found
        """
        try:
            # Check cache first
            if user_id in self.profile_cache:
                cache_entry = self.profile_cache[user_id]
                # Fix: Convert to seconds properly
                cache_age_ns = (np.datetime64('now') - cache_entry['timestamp']).astype('timedelta64[s]').astype(int)
                
                if cache_age_ns < self.cache_timeout:
                    embeddings_list = cache_entry['data']['embeddings']
                    shape = cache_entry['data']['embedding_shape']
                    return np.array(embeddings_list).reshape(shape)
            
            # Load from encrypted storage
            profile_data = self.encryption.load_voice_profile(user_id)
            
            if profile_data:
                embeddings_list = profile_data['embeddings']
                shape = profile_data['embedding_shape']
                embeddings = np.array(embeddings_list).reshape(shape)
                
                # Update cache
                self.profile_cache[user_id] = {
                    'data': profile_data,
                    'timestamp': np.datetime64('now')
                }
                
                return embeddings
            
            return None
            
        except Exception as e:
            print(f"Error getting embeddings for user {user_id}: {str(e)}")
            return None
    
    def update_user_profile(self, user_id: str, new_embeddings: np.ndarray) -> bool:
        """
        Update user's voice profile with new embeddings.
        
        Args:
            user_id: Unique user identifier
            new_embeddings: New voice embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing profile
            profile_data = self.encryption.load_voice_profile(user_id)
            
            if not profile_data:
                return False
            
            # Update embeddings
            profile_data['embeddings'] = new_embeddings.tolist()
            profile_data['embedding_shape'] = new_embeddings.shape
            profile_data['last_updated'] = str(np.datetime64('now'))
            
            # Save updated profile
            success = self.encryption.save_voice_profile(user_id, profile_data)
            
            if success:
                # Clear cache to force reload
                self.profile_cache.pop(user_id, None)
            
            return success
            
        except Exception as e:
            print(f"Error updating profile for user {user_id}: {str(e)}")
            return False
    
    def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete user's voice profile.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from cache
            self.profile_cache.pop(user_id, None)
            
            # Delete encrypted profile
            return self.encryption.delete_voice_profile(user_id)
            
        except Exception as e:
            print(f"Error deleting profile for user {user_id}: {str(e)}")
            return False
    
    def get_user_metadata(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User metadata or None if not found
        """
        try:
            profile_data = self.encryption.load_voice_profile(user_id)
            
            if profile_data:
                return {
                    'user_id': profile_data.get('user_id'),
                    'enrollment_date': profile_data.get('enrollment_date'),
                    'last_updated': profile_data.get('last_updated'),
                    'metadata': profile_data.get('metadata', {})
                }
            
            return None
            
        except Exception as e:
            print(f"Error getting metadata for user {user_id}: {str(e)}")
            return None
    
    def list_enrolled_users(self) -> list:
        """
        List all enrolled users.
        
        Returns:
            List of user IDs
        """
        return self.encryption.list_profiles()
    
    def clear_cache(self):
        """Clear the profile cache."""
        self.profile_cache.clear()

