"""
MFCC Extractor Module - Extracts Mel-Frequency Cepstral Coefficients.
Configured for optimal voice recognition: 32 MFCCs, 16kHz sample rate.
"""

import numpy as np
import librosa
from typing import Tuple


class MFCCExtractor:
    """
    MFCC feature extractor optimized for speaker verification.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mfcc: int = 32,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 64):
      
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Human voice frequency range
        self.fmin = 80   # Minimum frequency (Hz)
        self.fmax = 8000 # Maximum frequency (Hz)
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        """
        print(f" Extracting MFCC features...")
        
        # Validate input
        if len(audio) == 0:
            raise ValueError("Cannot extract MFCC from empty audio")
        
        if np.max(np.abs(audio)) < 1e-6:
            raise ValueError("Audio is too quiet for MFCC extraction")
        
        # Extract MFCC
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            # Transpose to (frames, n_mfcc) for CNN input
            mfcc = mfcc.T
            
            print(f" MFCC shape: {mfcc.shape} (frames={mfcc.shape[0]}, features={mfcc.shape[1]})")
            
            return mfcc
            
        except Exception as e:
            raise ValueError(f"MFCC extraction failed: {str(e)}")
    
    def extract_with_deltas(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC with delta and delta-delta features.
        
        
        """
        print(f"🎵 Extracting MFCC with deltas...")
        
        # Extract base MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Compute delta features
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        mfcc_combined = np.vstack([mfcc, delta, delta2])
        
        # Transpose to (frames, features)
        mfcc_combined = mfcc_combined.T
        
        print(f" MFCC+Deltas shape: {mfcc_combined.shape}")
        
        return mfcc_combined
    
    def normalize(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Normalize MFCC features 
        """
        mean = np.mean(mfcc, axis=0, keepdims=True)
        std = np.std(mfcc, axis=0, keepdims=True) + 1e-8
        
        mfcc_normalized = (mfcc - mean) / std
        
        return mfcc_normalized
    
    def pad_or_truncate(self, mfcc: np.ndarray, target_frames: int = 100) -> np.ndarray:
        """
        Pad or truncate MFCC to fixed number of frames.
        
        This ensures consistent input shape for the CNN.
        
        """
        current_frames = mfcc.shape[0]
        
        if current_frames < target_frames:
            # Pad with zeros
            pad_amount = target_frames - current_frames
            mfcc_padded = np.pad(mfcc, ((0, pad_amount), (0, 0)), mode='constant')
            return mfcc_padded
        
        elif current_frames > target_frames:
            # Truncate (take center)
            start = (current_frames - target_frames) // 2
            mfcc_truncated = mfcc[start:start + target_frames]
            return mfcc_truncated
        
        else:
            return mfcc
    
    def extract_for_cnn(self, audio: np.ndarray, target_frames: int = 100) -> np.ndarray:
        """
        Extract MFCC features ready for CNN input.
       
      
        """
        # Extract MFCC
        mfcc = self.extract(audio)
        
        # Normalize
        mfcc = self.normalize(mfcc)
        
        # Pad or truncate
        mfcc = self.pad_or_truncate(mfcc, target_frames)
        
        # Add channel dimension: (1, frames, features)
        mfcc_cnn = np.expand_dims(mfcc, axis=0)
        
        print(f" CNN-ready MFCC shape: {mfcc_cnn.shape}")
        
        return mfcc_cnn


def test_mfcc_extractor():
    """Test MFCC extraction."""
    print("=== MFCC Extractor Test ===")
    
    # Generate test audio (sine wave)
    sample_rate = 16000
    duration = 3.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Test basic extraction
    extractor = MFCCExtractor()
    
    print("\n1. Basic MFCC extraction:")
    mfcc = extractor.extract(audio)
    print(f"   Shape: {mfcc.shape}")
    print(f"   Mean: {np.mean(mfcc):.4f}")
    print(f"   Std: {np.std(mfcc):.4f}")
    
    print("\n2. MFCC with deltas:")
    mfcc_deltas = extractor.extract_with_deltas(audio)
    print(f"   Shape: {mfcc_deltas.shape}")
    
    print("\n3. Normalized MFCC:")
    mfcc_norm = extractor.normalize(mfcc)
    print(f"   Mean: {np.mean(mfcc_norm):.6f} (should be ~0)")
    print(f"   Std: {np.std(mfcc_norm):.6f} (should be ~1)")
    
    print("\n4. CNN-ready MFCC:")
    mfcc_cnn = extractor.extract_for_cnn(audio, target_frames=100)
    print(f"   Shape: {mfcc_cnn.shape}")
    
    print("\n All MFCC tests passed!")


if __name__ == "__main__":
    test_mfcc_extractor()
