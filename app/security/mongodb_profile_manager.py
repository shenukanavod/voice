"""
MongoDB Voice Profile Manager
Stores voice profiles in MongoDB Atlas instead of encrypted files

Flow: Audio → MFCC → CNN+LSTM → 128D Embeddings → MongoDB → Authentication
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine

from app.audio.preprocessing import AudioPreprocessor
from app.config import settings
from app.database.voice_db import VoiceDatabase
from model import CNNLSTMEmbedding  # CNN+LSTM model with MFCC
import librosa


class MongoDBVoiceProfileManager:
    """
    Voice profile manager that stores data in MongoDB.
    Uses CNN model for voice authentication with 128D embeddings.
    """

    def __init__(self, model_path: str = "models/fast_half_dataset_model.pth"):
        """Initialize MongoDB profile manager with CNN model.

        Args:
            model_path: Path to trained CNN model (default: fast_half_dataset_model.pth)
        """
        # Get MongoDB URL from environment or config
        mongodb_url = os.getenv("MONGODB_URL")

        # If not in environment, use from settings
        if not mongodb_url or mongodb_url == "":
            mongodb_url = settings.MONGODB_URL

        if not mongodb_url or mongodb_url == "":
            raise Exception(
                "❌ MongoDB URL not configured!\n"
                "Please set MONGODB_URL in .env file or app/config.py\n\n"
                "Example:\n"
                "MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/dbname"
            )

        # Try to initialize database connection with SSL error handling
        try:
            self.db = VoiceDatabase(
                connection_string=mongodb_url, database_name=settings.DATABASE_NAME
            )
        except Exception as e:
            error_str = str(e)
            if (
                "SSL" in error_str
                or "CERTIFICATE" in error_str
                or "certificate" in error_str
            ):
                print(f"⚠️ MongoDB SSL Certificate Error - attempting bypass...")
                # Try with SSL verification disabled
                if "tlsAllowInvalidCertificates=true" not in mongodb_url:
                    mongodb_url += "&tlsAllowInvalidCertificates=true"
                try:
                    self.db = VoiceDatabase(
                        connection_string=mongodb_url,
                        database_name=settings.DATABASE_NAME,
                    )
                    print("✅ MongoDB connected with SSL bypass")
                except:
                    raise Exception(f"MongoDB connection failed (SSL): {error_str}")
            else:
                raise e

        self.preprocessor = AudioPreprocessor()

        # MFCC configuration
        self.n_mfcc = settings.N_MFCC
        self.mfcc_feature_dim = self.n_mfcc * 3
        self.target_frames = 100

        # Load CNN+LSTM model (force CPU for compatibility with older GPUs)
        self.device = torch.device("cpu")
        self.model = None
        self.model_trained = False
        self.model_path = model_path

        self._load_model(model_path)

        print(f"✅ MongoDB Profile Manager initialized")
        print(f"   Database: {settings.DATABASE_NAME}")
        print(f"   Storage: Cloud (MongoDB Atlas)")
        print(f"   Model: CNN+LSTM ({model_path})")
        print(f"   Device: {self.device}")

    def _load_model(self, model_path: str):
        """Load CNN+LSTM model with configuration from checkpoint."""
        # Default values (fallback)
        embedding_dim = 256
        n_mels = 120

        if not Path(model_path).exists():
            print(f"⚠️  Model path not found: {model_path}")
            print("   Using untrained CNN+LSTM model (will train soon)")
            self.model = CNNLSTMEmbedding(
                embedding_dim=embedding_dim, n_mels=n_mels
            ).to(self.device)
            self.model.eval()
            self.model_trained = False
            return

        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Read config from checkpoint
            config = checkpoint.get('config', {})
            embedding_dim = config.get('embedding_dim', 256)
            n_mels = config.get('n_mels', 120)

            self.model = CNNLSTMEmbedding(
                embedding_dim=embedding_dim, n_mels=n_mels
            )

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "model_state" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.model_trained = True
            
            val_acc = checkpoint.get("val_accuracy", 0.0)
            epoch = checkpoint.get("epoch", 0)
            print(f"✅ Loaded trained CNN+LSTM model: {embedding_dim}D embeddings")
            print(f"   Training: Epoch {epoch}, Val Accuracy {val_acc:.2f}%")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Using untrained CNN+LSTM model")
            self.model = CNNLSTMEmbedding(
                embedding_dim=embedding_dim, n_mels=n_mels
            ).to(self.device)
            self.model.eval()
            self.model_trained = False

    def _prepare_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract and normalize MFCC features with fixed frame length."""
        mfcc_features = self.preprocessor.extract_mfcc(audio_data)

        mean = mfcc_features.mean()
        std = mfcc_features.std()
        if std > 1e-9:
            mfcc_features = (mfcc_features - mean) / std

        if mfcc_features.shape[1] < self.target_frames:
            pad_width = self.target_frames - mfcc_features.shape[1]
            mfcc_features = np.pad(
                mfcc_features, ((0, 0), (0, pad_width)), mode="constant"
            )
        elif mfcc_features.shape[1] > self.target_frames:
            mfcc_features = mfcc_features[:, : self.target_frames]

        return mfcc_features.astype(np.float32)

    def _extract_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC + Delta + Delta-Delta features"""
        # Extract 40 MFCC coefficients
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.preprocessor.sample_rate, 
                                     n_mfcc=40, n_fft=2048, hop_length=512)
        
        # Compute deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack to get 120 features (40 + 40 + 40)
        features = np.vstack([mfcc, delta, delta2])
        
        # Pad or truncate to 100 frames
        if features.shape[1] < 100:
            features = np.pad(features, ((0, 0), (0, 100 - features.shape[1])), mode='constant')
        else:
            features = features[:, :100]
        
        return features.astype(np.float32)

    def _extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract embedding using CNN+LSTM model with MFCC features."""
        
        # Normalize audio
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Clip to [-1, 1]
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Extract MFCC features
        mfcc_features = self._extract_mfcc(audio_data)

        print("🔍 Extracting embedding: model=CNN+LSTM (MFCC)")
        print(
            f"   Audio data: shape={audio_data.shape}, range=[{audio_data.min():.4f}, {audio_data.max():.4f}]"
        )
        print(f"   MFCC features: shape={mfcc_features.shape}")

        if self.model is None:
            raise Exception("CNN+LSTM model not initialized")

        # Convert to tensor: (120, 100) → (1, 1, 120, 100) for 2D Conv
        mfcc_tensor = torch.from_numpy(mfcc_features).float().unsqueeze(0).unsqueeze(0)
        mfcc_tensor = mfcc_tensor.to(self.device)

        with torch.no_grad():
            embedding = self.model(mfcc_tensor)

        result = embedding.squeeze(0).cpu().numpy()
        print(
            f"   Extracted embedding: shape={result.shape}, norm={np.linalg.norm(result):.4f}"
        )

        return result

    def create_profile(
        self,
        user_id: str,
        audio_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create voice profile from audio and save to MongoDB.

        Flow: Audio → MFCC → CNN+LSTM → 128D Embedding → MongoDB

        Args:
            user_id: Unique user identifier
            audio_data: Preprocessed audio array
            metadata: Optional metadata

        Returns:
            Profile data dictionary
        """
        try:
            # Extract embedding using CNN or MFCC stats
            embeddings = self._extract_embedding(audio_data)

            model_type = "CNN-LSTM"
            if not self.model_trained:
                model_type = "CNN-LSTM (untrained)"
            print(f"✅ Extracted {model_type}: {embeddings.shape}")

            # Save to MongoDB
            success = self.db.save_embedding(user_id, embeddings)

            if success:
                profile_data = {
                    "user_id": user_id,
                    "embeddings": embeddings.tolist(),
                    "embedding_shape": embeddings.shape,
                    "model_type": model_type,
                    "metadata": metadata or {},
                }
                return profile_data
            else:
                raise Exception("Failed to save profile to MongoDB")

        except Exception as e:
            raise Exception(f"Error creating profile for user {user_id}: {str(e)}")

    def extract_embedding_from_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Public helper to extract a single embedding from preprocessed audio."""
        return self._extract_embedding(audio_data)

    def create_profile_from_embeddings(
        self,
        user_id: str,
        embeddings: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create voice profile from a precomputed embedding and save to MongoDB.

        Flow: Embedding → MongoDB
        """
        try:
            success = self.db.save_embedding(user_id, embeddings)

            if success:
                profile_data = {
                    "user_id": user_id,
                    "embeddings": embeddings.tolist(),
                    "embedding_shape": embeddings.shape,
                    "model_type": "CNN-LSTM",
                    "metadata": metadata or {},
                }
                return profile_data
            else:
                raise Exception("Failed to save profile to MongoDB")

        except Exception as e:
            raise Exception(
                f"Error creating profile from embeddings for user {user_id}: {str(e)}"
            )

    def verify(self, user_id: str, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Verify user's voice against stored profile in MongoDB.

        Flow: Audio → MFCC → CNN+LSTM → Embedding → Compare with MongoDB → Authenticate

        Args:
            user_id: Unique user identifier
            audio_data: Preprocessed audio array to verify

        Returns:
            Dictionary with verification result
        """
        try:
            # Get stored embedding from MongoDB
            stored_embedding = self.db.get_embedding(user_id)

            if stored_embedding is None:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "user_id": user_id,
                    "error": "User profile not found in MongoDB",
                }

            # Extract embedding from new audio using CNN+LSTM
            new_embedding = self._extract_embedding(audio_data)

            # Ensure same length
            min_len = min(len(new_embedding), len(stored_embedding))
            new_embedding = new_embedding[:min_len]
            stored_embedding = stored_embedding[:min_len]

            # Calculate cosine similarity
            similarity = 1 - cosine(new_embedding, stored_embedding)

            threshold = settings.VERIFICATION_THRESHOLD

            verified = similarity >= threshold

            model_type = "CNN-LSTM"
            print(f"🔍 MongoDB Verification ({model_type}): {user_id}")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Threshold: {threshold:.3f}")
            print(f"   Result: {'✅ VERIFIED' if verified else '❌ REJECTED'}")

            return {
                "verified": verified,
                "similarity": float(similarity),
                "user_id": user_id,
                "threshold": threshold,
                "model_type": model_type,
                "storage": "MongoDB",
                "stored_embedding": stored_embedding,  # Include for monitoring
            }

        except Exception as e:
            print(f"❌ Verification error: {str(e)}")
            return {
                "verified": False,
                "similarity": 0.0,
                "user_id": user_id,
                "error": str(e),
            }

    def enroll_user(
        self,
        user_id: str,
        voice_embeddings: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Enroll user with voice embeddings in MongoDB.

        Args:
            user_id: Unique user identifier
            voice_embeddings: Voice embeddings array
            metadata: Optional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.db.save_embedding(user_id, voice_embeddings)

            if success:
                print(f"✅ User {user_id} enrolled in MongoDB")

            return success

        except Exception as e:
            print(f"❌ Error enrolling user {user_id}: {str(e)}")
            return False

    def get_user_embeddings(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get voice embeddings for a user from MongoDB.

        Args:
            user_id: Unique user identifier

        Returns:
            Voice embeddings array or None if not found
        """
        return self.db.get_embedding(user_id)

    def user_exists(self, user_id: str) -> bool:
        """
        Check if user exists in MongoDB.

        Args:
            user_id: User identifier

        Returns:
            True if user exists, False otherwise
        """
        return self.db.user_exists(user_id)

    def delete_user(self, user_id: str) -> bool:
        """
        Delete user profile from MongoDB.

        Args:
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        return self.db.delete_user(user_id)

    def get_all_users(self):
        """Get list of all enrolled users from MongoDB."""
        return self.db.get_all_users()

    def count_users(self) -> int:
        """Count total enrolled users in MongoDB."""
        return self.db.count_users()

    def close(self):
        """Close database connection."""
        self.db.close()


if __name__ == "__main__":
    # Test MongoDB profile manager
    print("\n" + "=" * 60)
    print("TESTING MONGODB PROFILE MANAGER")
    print("=" * 60 + "\n")

    try:
        manager = MongoDBVoiceProfileManager()

        # Test with dummy audio
        dummy_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1

        # Test create profile
        print("Test 1: Create profile...")
        profile = manager.create_profile("test_user_001", dummy_audio)
        print(f"   Created: {profile['user_id']}")

        # Test verify
        print("\nTest 2: Verify user...")
        result = manager.verify("test_user_001", dummy_audio)
        print(f"   Verified: {result['verified']}")
        print(f"   Similarity: {result['similarity']:.3f}")

        # Test user exists
        print("\nTest 3: Check user exists...")
        exists = manager.user_exists("test_user_001")
        print(f"   Exists: {exists}")

        # Test count users
        print("\nTest 4: Count users...")
        count = manager.count_users()
        print(f"   Total users: {count}")

        # Cleanup
        print("\nTest 5: Delete user...")
        deleted = manager.delete_user("test_user_001")
        print(f"   Deleted: {deleted}")

        manager.close()

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
