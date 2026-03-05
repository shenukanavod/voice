"""
Authentication Module
Login authentication: record audio, extract embedding, compare with stored, authenticate
"""

import numpy as np
from typing import Tuple, Dict, Any
from scipy.spatial.distance import cosine
from audio_recording import VoiceRecorder
from preprocessing import AudioPreprocessor
from embedding import EmbeddingExtractor
from db import VoiceDatabase
from app.config import settings


class AuthenticationService:
    """Complete authentication workflow."""

    def __init__(self, model_path: str = "models/fast_half_dataset_model.pth"):
        self.recorder = VoiceRecorder(
            sample_rate=settings.SAMPLE_RATE,
            duration=settings.AUDIO_DURATION,
        )
        self.preprocessor = AudioPreprocessor(sample_rate=settings.SAMPLE_RATE)
        self.embedding_extractor = EmbeddingExtractor(
            model_path=model_path, device="cpu"
        )

        # MongoDB
        mongodb_url = settings.MONGODB_URL
        if "tlsAllowInvalidCertificates=true" not in mongodb_url:
            mongodb_url += "&tlsAllowInvalidCertificates=true"

        self.db = VoiceDatabase(
            connection_string=mongodb_url, database_name=settings.DATABASE_NAME
        )

        self.threshold = settings.VERIFICATION_THRESHOLD

    def authenticate_user(
        self, user_id: str, custom_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Authenticate a user by voice.

        Args:
            user_id: User to authenticate
            custom_threshold: Optional custom similarity threshold

        Returns:
            Dictionary with authentication result
        """
        threshold = custom_threshold or self.threshold

        print(f"\n{'='*60}")
        print(f"AUTHENTICATION - User: {user_id}")
        print(f"{'='*60}\n")

        # Check if user exists
        if not self.db.user_exists(user_id):
            print(f"❌ User {user_id} not enrolled")
            return {
                "authenticated": False,
                "user_id": user_id,
                "similarity": 0.0,
                "message": f"User {user_id} not enrolled",
            }

        # Load stored embedding
        print(f"Loading stored embedding for {user_id}...")
        stored_embedding = self.db.get_embedding(user_id)

        if stored_embedding is None:
            print(f"❌ Failed to load embedding")
            return {
                "authenticated": False,
                "user_id": user_id,
                "similarity": 0.0,
                "message": "Failed to load stored embedding",
            }

        print(f"✅ Loaded stored embedding: shape {stored_embedding.shape}")

        # Record authentication sample
        try:
            print(f"\nRecording authentication sample...")
            audio = self.recorder.record()

            # Preprocess
            print(f"Preprocessing...")
            audio = self.preprocessor.preprocess(audio)

            # Extract embedding
            print(f"Extracting embedding...")
            test_embedding = self.embedding_extractor.extract(audio)

            print(f"✅ Test embedding: shape {test_embedding.shape}")

        except Exception as e:
            print(f"❌ Error during recording/processing: {e}")
            return {
                "authenticated": False,
                "user_id": user_id,
                "similarity": 0.0,
                "message": f"Error during recording: {str(e)}",
            }

        # Compute similarity
        print(f"\nComputing similarity...")

        # Cosine similarity
        similarity = 1 - cosine(stored_embedding, test_embedding)

        print(f"Similarity: {similarity:.6f}")
        print(f"Threshold: {threshold:.6f}")

        # Authenticate
        authenticated = similarity >= threshold

        print(f"\n{'='*40}")
        if authenticated:
            print(f"✅ AUTHENTICATED - Welcome {user_id}!")
        else:
            print(f"❌ NOT AUTHENTICATED - Access denied")
        print(f"{'='*40}\n")

        return {
            "authenticated": authenticated,
            "user_id": user_id,
            "similarity": float(similarity),
            "threshold": threshold,
            "message": "Authenticated" if authenticated else "Not authenticated",
        }

    def close(self):
        """Close database connection."""
        self.db.close()


if __name__ == "__main__":
    service = AuthenticationService()

    user_id = input("Enter user ID: ").strip()
    if not user_id:
        print("❌ User ID required")
        exit(1)

    result = service.authenticate_user(user_id)

    print(f"\n{'='*60}")
    print(f"Result: {result['message']}")
    print(f"Similarity: {result['similarity']:.6f}")
    print(f"{'='*60}\n")

    service.close()
