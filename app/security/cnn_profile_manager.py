"""
CNN-based Voice Profile Manager
Uses trained speaker embedding model for 128-D embeddings
Better accuracy than MFCC-based approach
"""

from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine

from app.config import settings
from app.security.encryption import VoiceProfileEncryption


class CNNLSTMEmbedding(nn.Module):
    """CNN-LSTM model for speaker embedding extraction."""
    
def __init__(self, embedding_dim: int = 256, n_mels: int = 120):
        super(CNNLSTMEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # LSTM
        self.lstm_input_size = 256 * (n_mels // 8)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted = lstm_out * attention_weights
        pooled = weighted.sum(dim=1)
        
        # Embeddings
        embeddings = self.embedding(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class CNNVoiceProfileManager:
    """
    High-level manager using CNN embeddings for voice profiles.
    Uses trained speaker_embedding_model.pth
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize CNN-based profile manager.

        Args:
            model_path: Path to trained model (default: speaker_verification/models/speaker_embedding_model.pth)
        """
        # SET DETERMINISTIC SEED FOR REPRODUCIBILITY
        # This ensures that if model loading fails and falls back to untrained model,
        # the model will have the same weights each time (deterministic)
        self._set_seeds(seed=42)
        
        self.encryption = VoiceProfileEncryption()
        self.device = torch.device("cpu")  # Use CPU for inference

        # Model parameters
        self.sample_rate = 16000
        self.n_mels = 120
        self.n_fft = 2048
        self.hop_length = 512
        self.duration = 3.0

        # Load trained model
        if model_path is None:
            model_path = "speaker_verification/models/speaker_embedding_model.pth"

        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode

        print(f"CNN Model loaded from {model_path}")
        print(f"Embedding dimension: 128")
        print(f"Device: {self.device}")
    
    @staticmethod
    def _set_seeds(seed: int = 42):
        """Set all random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained CNN-LSTM model."""
        try:
            # Try to load checkpoint to get config first
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Read config from checkpoint
                config = checkpoint.get('config', {})
                embedding_dim = config.get('embedding_dim', 256)
                n_mels = config.get('n_mels', 120)
            else:
                # Use defaults if file doesn't exist
                embedding_dim = 256
                n_mels = 120
                checkpoint = None
            
            model = CNNLSTMEmbedding(embedding_dim=embedding_dim, n_mels=n_mels)

            if checkpoint is not None:
                # Handle different checkpoint formats
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"   ✅ Loaded trained model from epoch {checkpoint.get('epoch', 'unknown')}")
                    if 'val_accuracy' in checkpoint:
                        print(f"   Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
                elif "model_state" in checkpoint:
                    # Triplet loss training format
                    model.load_state_dict(checkpoint["model_state"])
                    print(f"   ✅ Loaded triplet model from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"   ✅ Loaded model weights")

                model.to(self.device)
                print(f"   Model is TRAINED and ready")
                return model
            else:
                print(f"⚠️  Model file not found: {model_path}")
                print(f"   Using UNTRAINED model - authentication will NOT work properly!")
                return model.to(self.device)

        except Exception as e:
            print(f"❌ Model loading error: {type(e).__name__}: {e}")
            print(f"   Using UNTRAINED model - authentication will NOT work properly!")
            model = CNNLSTMEmbedding(embedding_dim=128, n_mels=self.n_mels)
            return model.to(self.device)

    def _audio_to_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to log-mel spectrogram.

        Args:
            audio: Audio array

        Returns:
            Log-mel spectrogram (n_mels, time_frames)
        """
        # Ensure correct length
        target_length = int(self.sample_rate * self.duration)
        if len(audio) > target_length:
            start = (len(audio) - target_length) // 2
            audio = audio[start : start + target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")

        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return log_mel_spec

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract 128-D embedding from audio using CNN model.

        Args:
            audio: Preprocessed audio array

        Returns:
            128-dimensional embedding vector
        """
        try:
            # Convert to mel spectrogram
            mel_spec = self._audio_to_melspectrogram(audio)

            # Convert to tensor and add batch + channel dimensions
            mel_tensor = (
                torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
            )  # (1, 1, n_mels, time)
            mel_tensor = mel_tensor.to(self.device)

            # Extract embedding
            with torch.no_grad():
                embedding = self.model(mel_tensor)

            # Convert to numpy
            embedding = embedding.cpu().numpy().flatten()

            return embedding

        except Exception as e:
            raise Exception(f"Error extracting embedding: {str(e)}")

    def create_profile(
        self,
        user_id: str,
        audio_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create voice profile using CNN embeddings.

        Args:
            user_id: Unique user identifier
            audio_data: Preprocessed audio array
            metadata: Optional metadata

        Returns:
            Profile data dictionary
        """
        try:
            # Extract 128-D embedding
            embedding = self._extract_embedding(audio_data)

            print(
                f"✅ Extracted embedding: shape {embedding.shape}, norm {np.linalg.norm(embedding):.3f}"
            )

            # Create profile data
            profile_data = {
                "user_id": user_id,
                "embeddings": embedding.tolist(),
                "embedding_dim": len(embedding),
                "model_type": "CNN",
                "enrollment_date": str(np.datetime64("now")),
                "metadata": metadata or {},
            }

            # Save encrypted profile
            success = self.encryption.save_voice_profile(user_id, profile_data)

            if success:
                print(f"✅ Profile saved for user {user_id}")
                return profile_data
            else:
                raise Exception("Failed to save profile")

        except Exception as e:
            raise Exception(f"Error creating profile for user {user_id}: {str(e)}")

    def verify(self, user_id: str, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Verify user's voice against stored CNN embedding.

        Args:
            user_id: Unique user identifier
            audio_data: Preprocessed audio array to verify

        Returns:
            Dictionary with verification result
        """
        try:
            # Load stored profile
            profile_data = self.encryption.load_voice_profile(user_id)

            if not profile_data:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "user_id": user_id,
                    "error": "User profile not found",
                }

            # Extract embedding from new audio
            new_embedding = self._extract_embedding(audio_data)

            # Get stored embedding
            stored_embedding = np.array(profile_data["embeddings"])

            # Ensure L2 normalized
            new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
            stored_embedding = stored_embedding / (
                np.linalg.norm(stored_embedding) + 1e-8
            )

            # Check for near-zero embeddings (potential error)
            if (
                np.max(np.abs(new_embedding)) < 0.01
                or np.max(np.abs(stored_embedding)) < 0.01
            ):
                print(f"⚠️  Warning: Very small embedding values detected")
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "user_id": user_id,
                    "error": "Invalid embedding values",
                }

            # Check embedding diversity (should have variation)
            new_std = np.std(new_embedding)
            stored_std = np.std(stored_embedding)
            if new_std < 0.05 or stored_std < 0.05:
                print(
                    f"⚠️  Warning: Embedding lacks diversity (std: {new_std:.4f}, {stored_std:.4f})"
                )
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "user_id": user_id,
                    "error": "Low quality audio or invalid voice",
                }

            # Calculate cosine similarity
            similarity = 1 - cosine(new_embedding, stored_embedding)

            # STRICT threshold for verification (80% minimum)
            # Plus additional embedding quality checks above
            threshold = 0.80
            verified = similarity >= threshold

            # Additional check: Embeddings should not be suspiciously identical
            # (could indicate model collapse or testing with same audio)
            if verified and similarity > 0.998:
                print(f"⚠️  Warning: Suspiciously high similarity ({similarity:.4f})")
                # Check if embeddings are actually different
                embedding_diff = np.abs(new_embedding - stored_embedding).mean()
                if embedding_diff < 0.001:
                    print(
                        f"⚠️  Warning: Embeddings are nearly identical (diff: {embedding_diff:.6f})"
                    )
                    print(f"⚠️  This may be the same audio used for registration")

            print(f"🔍 Verification: {user_id}")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Threshold: {threshold:.3f}")
            print(f"   Result: {'✅ VERIFIED' if verified else '❌ REJECTED'}")

            return {
                "verified": verified,
                "similarity": float(similarity),
                "user_id": user_id,
                "threshold": threshold,
                "model_type": "CNN",
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

    def delete_profile(self, user_id: str) -> bool:
        """Delete user profile."""
        return self.encryption.delete_voice_profile(user_id)

    # --- Compatibility helpers expected by desktop_app.py ---
    def user_exists(self, user_id: str) -> bool:
        """Check if a user profile exists in encrypted storage."""
        data = self.encryption.load_voice_profile(user_id)
        return data is not None

    def get_user_embeddings(self, user_id: str) -> Optional[np.ndarray]:
        """Return stored embeddings for a user, or None if not found."""
        data = self.encryption.load_voice_profile(user_id)
        if not data:
            return None
        embeddings = np.array(data.get("embeddings"))
        return embeddings

    def enroll_user(self, user_id: str, embeddings: np.ndarray) -> bool:
        """Save or update a user's embeddings in encrypted storage.

        The desktop UI passes precomputed embeddings; we persist them.
        """
        try:
            profile_data = {
                "user_id": user_id,
                "embeddings": np.asarray(embeddings).flatten().tolist(),
                "embedding_dim": int(np.asarray(embeddings).size),
                "model_type": "CNN",
                "enrollment_date": str(np.datetime64("now")),
                "metadata": {"source": "desktop_app", "method": "enroll_user"},
            }
            return self.encryption.save_voice_profile(user_id, profile_data)
        except Exception as e:
            print(f"❌ Failed to enroll user {user_id}: {e}")
            return False

    def profile_exists(self, user_id: str) -> bool:
        """Check if user profile exists."""
        profile_file = Path(settings.VOICE_PROFILES_PATH) / f"{user_id}.enc"
        return profile_file.exists()


# Singleton instance for easy import
_cnn_manager_instance = None


def get_cnn_profile_manager() -> CNNVoiceProfileManager:
    """Get singleton instance of CNN profile manager."""
    global _cnn_manager_instance
    if _cnn_manager_instance is None:
        _cnn_manager_instance = CNNVoiceProfileManager()
    return _cnn_manager_instance


if __name__ == "__main__":
    # Test the CNN profile manager
    print("Testing CNN Profile Manager...")

    manager = CNNVoiceProfileManager()

    # Test with dummy audio
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)

    # Test embedding extraction
    embedding = manager._extract_embedding(dummy_audio)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.3f}")
