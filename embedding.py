"""
Embedding Module
Extract embeddings from audio using CNN+LSTM model
"""

import torch
import numpy as np
from typing import Optional
from preprocessing import AudioPreprocessor
from model import CNNLSTMEmbedding, load_model


class EmbeddingExtractor:
    """Extract embeddings from audio using CNN+LSTM (auto-detects embedding dimension from model)."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.preprocessor = AudioPreprocessor()

        self.model, self.is_trained = load_model(model_path, device=device)
        self.embedding_dim = self.model.embedding_dim

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract embedding from raw audio.

        Args:
            audio: Raw audio array (float32)

        Returns:
            Embedding vector (L2 normalized, dimension depends on model)
        """
        # Preprocess audio
        audio = self.preprocessor.preprocess(audio)

        # Extract MFCC
        mfcc = self.preprocessor.extract_mfcc(audio)

        # Pad MFCC to fixed size
        mfcc = self.preprocessor.pad_mfcc(mfcc, target_frames=100)

        # Convert to tensor
        mfcc_tensor = (
            torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # Extract embedding
        with torch.no_grad():
            embedding = self.model(mfcc_tensor)

        # Convert to numpy
        embedding = embedding.squeeze(0).cpu().numpy()

        return embedding

    def extract_batch(self, audio_list: list) -> np.ndarray:
        """
        Extract embeddings from multiple audio samples.

        Args:
            audio_list: List of audio arrays

        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        embeddings = []
        for audio in audio_list:
            embedding = self.extract(audio)
            embeddings.append(embedding)

        return np.vstack(embeddings)
