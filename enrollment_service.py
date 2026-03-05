"""
Enrollment Pipeline Module
Clean enrollment logic: record 3 samples, extract embeddings, average, save to MongoDB
"""

import numpy as np
from typing import Tuple
from audio_recording import VoiceRecorder
from preprocessing import AudioPreprocessor
from embedding import EmbeddingExtractor
from db import VoiceDatabase
from app.config import settings


class EnrollmentService:
    """Complete enrollment workflow."""

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

    def enroll_user(
        self, user_id: str, num_samples: int = 3
    ) -> Tuple[bool, str, np.ndarray]:
        """
        Enroll a user with multiple voice samples.

        Args:
            user_id: Unique user identifier
            num_samples: Number of samples to record (default 3)

        Returns:
            (success, message, final_embedding)
        """
        print(f"\n{'='*60}")
        print(f"ENROLLMENT - User: {user_id}")
        print(f"{'='*60}\n")

        embeddings_list = []

        for i in range(num_samples):
            print(f"\n📝 Sample {i+1}/{num_samples}")
            print(f"{'='*40}")

            try:
                # Record
                print(f"Recording...")
                audio = self.recorder.record()

                # Preprocess
                print(f"Preprocessing...")
                audio = self.preprocessor.preprocess(audio)

                # Extract embedding
                print(f"Extracting embedding...")
                embedding = self.embedding_extractor.extract(audio)

                embeddings_list.append(embedding)
                print(f"✅ Sample {i+1} processed: embedding shape {embedding.shape}")

            except Exception as e:
                return (False, f"Error during sample {i+1}: {str(e)}", None)

        # Average embeddings
        print(f"\n{'='*40}")
        print(f"Averaging {num_samples} embeddings...")

        embeddings_array = np.vstack(embeddings_list)
        final_embedding = np.mean(embeddings_array, axis=0)

        # L2 normalize
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm

        print(f"✅ Final embedding: shape {final_embedding.shape}, norm {norm:.6f}")

        # Save to MongoDB
        print(f"\nSaving to MongoDB...")
        success = self.db.save_embedding(user_id, final_embedding)

        if success:
            print(f"\n✅ User {user_id} enrolled successfully!")
            return (True, f"User {user_id} enrolled with {num_samples} samples", final_embedding)
        else:
            return (False, "Failed to save profile to MongoDB", None)

    def close(self):
        """Close database connection."""
        self.db.close()
