
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio.voice_capture import VoiceCapture
from app.audio.noise_filter import NoiseFilter
from app.audio.vad_processor import VADProcessor
from app.audio.mfcc_extractor import MFCCExtractor
from app.models.cnn_embedding import EmbeddingExtractor


class EnrollmentPipeline:
    """
    Complete enrollment pipeline for voice authentication.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 duration: float = 3.0,
                 model_path: Optional[str] = None):
        """
        Initialize enrollment pipeline.
        
  
        """
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Initialize components
        self.voice_capture = VoiceCapture(sample_rate=sample_rate, duration=duration)
        self.noise_filter = NoiseFilter(sample_rate=sample_rate)
        self.vad_processor = VADProcessor(sample_rate=sample_rate)
        self.mfcc_extractor = MFCCExtractor(sample_rate=sample_rate)
        self.embedding_extractor = EmbeddingExtractor(model_path=model_path)
        
        print("Enrollment pipeline initialized")
    
    def enroll_user(self, user_id: str) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Enroll a user by recording voice and generating embedding.
        
        """
        print(f"\n{'='*60}")
        print(f"🎤 VOICE ENROLLMENT - User: {user_id}")
        print(f"{'='*60}\n")
        
        try:
            # STEP 1: Record audio
            print("STEP 1/7: Recording audio...")
            audio, is_valid, error = self.voice_capture.record_audio()
            
            if not is_valid:
                return False, None, f"Recording failed: {error}"
            
            print(f" Recorded {len(audio)/self.sample_rate:.2f}s of audio\n")
            
            # STEP 2: Validate human voice (RMS check already done in capture)
            print("STEP 2/7: Validating human voice...")
            
            # Check if audio contains human speech frequencies
            is_speech, error = self._validate_human_voice(audio)
            if not is_speech:
                return False, None, error
            
            print(" Human voice detected\n")
            
            # STEP 3: Remove noise
            print("STEP 3/7: Removing background noise...")
            audio_clean = self.noise_filter.remove_noise(audio)
            audio_clean = self.noise_filter.apply_bandpass_filter(audio_clean, lowcut=80, highcut=3400)
            print(f" Noise removed\n")
            
            # STEP 4: Apply VAD to extract speech segments
            print("STEP 4/7: Extracting speech segments (VAD)...")
            audio_speech, voiced_percentage = self.vad_processor.detect_voice(audio_clean)
            
            if voiced_percentage < 30.0:
                return False, None, f"Insufficient speech detected. Only {voiced_percentage:.1f}% voiced frames (minimum: 30%). Please speak clearly."
            
            if len(audio_speech) < int(self.sample_rate * 0.3):
                return False, None, "Speech too brief. Please speak for at least half a second."
            
            print(f" Extracted {len(audio_speech)/self.sample_rate:.2f}s of speech ({voiced_percentage:.1f}% voiced)\n")
            
            # STEP 5: Extract MFCC features
            print("STEP 5/7: Extracting MFCC features...")
            mfcc = self.mfcc_extractor.extract_for_cnn(audio_speech, target_frames=100)
            print(f" MFCC shape: {mfcc.shape}\n")
            
            # STEP 6: Generate embedding
            print("STEP 6/7: Generating speaker embedding...")
            embedding = self.embedding_extractor.extract_embedding(mfcc)
            
            # Verify embedding is normalized
            norm = np.linalg.norm(embedding)
            print(f" Embedding generated: shape={embedding.shape}, norm={norm:.6f}\n")
            
            if norm < 0.9 or norm > 1.1:
                print(f"⚠️ Warning: Embedding norm is {norm:.6f} (expected ~1.0)")
            
            # STEP 7: Success
            print("STEP 7/7: Enrollment complete!")
            print(f"{'='*60}")
            print(f" SUCCESS - User '{user_id}' enrolled successfully")
            print(f"   Embedding: 128-dimensional vector (norm={norm:.6f})")
            print(f"{'='*60}\n")
            
            return True, embedding, "Enrollment successful"
            
        except Exception as e:
            error_msg = f"Enrollment failed: {str(e)}"
            print(f"\n {error_msg}\n")
            return False, None, error_msg
    
    def _validate_human_voice(self, audio: np.ndarray) -> Tuple[bool, str]:
        """
        Validate that audio contains human voice (not noise/music/tapping).
     
        """
        # Compute FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Total energy
        total_energy = np.sum(magnitude ** 2)
        
        if total_energy < 1e-10:
            return False, "No audio signal detected"
        
        # Human voice frequency range: 80-3400 Hz
        voice_mask = (freqs >= 80) & (freqs <= 3400)
        voice_energy = np.sum(magnitude[voice_mask] ** 2)
        
        # Calculate ratio
        voice_ratio = voice_energy / total_energy
        
        print(f" Voice frequency energy: {voice_ratio:.2%}")
        
        # Require at least 10% of energy in voice range
        if voice_ratio < 0.10:
            return False, f"No human voice detected (voice energy: {voice_ratio:.2%}). Only background noise present. Please speak clearly into the microphone."
        
        return True, ""
    
    def close(self):
        """Close resources."""
        self.voice_capture.close()


def main():
    """Test enrollment pipeline."""
    print("\n" + "="*60)
    print("VOICE AUTHENTICATION - ENROLLMENT TEST")
    print("="*60 + "\n")
    
    # Initialize pipeline
    pipeline = EnrollmentPipeline()
    
    try:
        # Test enrollment
        user_id = "test_user_001"
        success, embedding, message = pipeline.enroll_user(user_id)
        
        if success:
            print(f"\n Enrollment successful!")
            print(f"   User ID: {user_id}")
            print(f"   Embedding shape: {embedding.shape}")
            print(f"   Embedding norm: {np.linalg.norm(embedding):.6f}")
            print(f"   First 10 values: {embedding[:10]}")
        else:
            print(f"\n Enrollment failed: {message}")
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
