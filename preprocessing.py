"""
Preprocessing Module
Audio preprocessing: VAD, normalization, MFCC extraction
"""

import numpy as np
import librosa
from typing import Tuple
from app.config import settings


class AudioPreprocessor:
    """Complete audio preprocessing pipeline."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.n_mfcc = settings.N_MFCC
        self.n_fft = settings.N_FFT
        self.hop_length = settings.HOP_LENGTH
        self.n_mels = settings.N_MELS

    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing: VAD, normalization, ensure correct length.

        Args:
            audio: Raw audio array

        Returns:
            Preprocessed audio (float32, normalized)
        """
        # Validate input
        if len(audio) == 0:
            raise ValueError("Audio array is empty")

        # Auto-gain control
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0.000001:
            target_rms = 0.35
            gain = target_rms / rms
            gain = min(max(gain, 0.5), 200.0)
            audio = audio * gain

        # Simple VAD: remove leading/trailing silence
        audio = self._trim_silence(audio)

        # Normalize
        audio = np.clip(audio, -1.0, 1.0)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        # Ensure correct length
        target_length = int(self.sample_rate * settings.AUDIO_DURATION)
        if len(audio) > target_length:
            start = (len(audio) - target_length) // 2
            audio = audio[start : start + target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")

        return audio.astype(np.float32)

    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.00005) -> np.ndarray:
        """Remove leading and trailing silence."""
        rms = np.sqrt(np.mean(audio**2))
        if rms < threshold:
            return audio

        # Find start
        cumsum = np.cumsum(np.abs(audio))
        start = np.argmax(cumsum > cumsum[-1] * 0.01)

        # Find end
        end = len(audio) - np.argmax(cumsum[::-1] > cumsum[-1] * 0.01)

        return audio[start:end]

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio: Preprocessed audio

        Returns:
            MFCC features (n_mfcc * 3, time_frames) with delta and delta-delta
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Add delta features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # Stack
        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

        # Normalize
        mean = features.mean()
        std = features.std()
        if std > 1e-9:
            features = (features - mean) / std

        return features.astype(np.float32)

    def pad_mfcc(
        self, mfcc: np.ndarray, target_frames: int = 100
    ) -> np.ndarray:
        """Pad or truncate MFCC to fixed number of frames."""
        current_frames = mfcc.shape[1]

        if current_frames < target_frames:
            pad_width = target_frames - current_frames
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        elif current_frames > target_frames:
            mfcc = mfcc[:, :target_frames]

        return mfcc
