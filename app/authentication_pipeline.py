"""
Authentication Pipeline - Complete voice authentication/login system.

Pipeline:
1. Record audio
2. Validate human voice (same as enrollment)
3. Remove noise
4. Extract MFCC
5. Generate CNN embedding
6. Compare with stored embedding using cosine similarity
7. Authenticate if similarity ≥ 0.80
"""

import numpy as np
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


class AuthenticationPipeline:
    """
    Complete authentication pipeline for voice verification.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 duration: float = 3.0,
                 model_path: Optional[str] = None,
                 threshold: float = 0.80):
        """
        Initialize authentication pipeline.
        
       
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.threshold = threshold
        
        # Initialize components (same as enrollment)
        self.voice_capture = VoiceCapture(sample_rate=sample_rate, duration=duration)
        self.noise_filter = NoiseFilter(sample_rate=sample_rate)
        self.vad_processor = VADProcessor(sample_rate=sample_rate)
        self.mfcc_extractor = MFCCExtractor(sample_rate=sample_rate)
        self.embedding_extractor = EmbeddingExtractor(model_path=model_path)
        
        print(" Authentication pipeline initialized")
        print(f"   Similarity threshold: {self.threshold:.2f} (≥{self.threshold:.0%} required)")
    
    def authenticate_user(self, 
                         user_id: str, 
                         stored_embedding: np.ndarray) -> Tuple[bool, float, str]:
        """
        Authenticate a user by comparing live voice to stored embedding.
        
        """
        print(f"\n{'='*60}")
        print(f"VOICE AUTHENTICATION - User: {user_id}")
        print(f"{'='*60}\n")
        
        try:
            # STEP 1: Record audio
            print("STEP 1/6: Recording audio...")
            audio, is_valid, error = self.voice_capture.record_audio()
            
            if not is_valid:
                return False, 0.0, f"Recording failed: {error}"
            
            print(f" Recorded {len(audio)/self.sample_rate:.2f}s of audio\n")
            
            # STEP 2: Validate human voice
            print("STEP 2/6: Validating human voice...")
            is_speech, error = self._validate_human_voice(audio)
            if not is_speech:
                return False, 0.0, error
            
            print(" Human voice detected\n")
            
            # STEP 3: Remove noise
            print("STEP 3/6: Removing background noise...")
            audio_clean = self.noise_filter.remove_noise(audio)
            audio_clean = self.noise_filter.apply_bandpass_filter(audio_clean, lowcut=80, highcut=3400)
            print(f" Noise removed\n")
            
            # STEP 4: Apply VAD
            print("STEP 4/6: Extracting speech segments (VAD)...")
            audio_speech, voiced_percentage = self.vad_processor.detect_voice(audio_clean)
            
            if voiced_percentage < 30.0:
                return False, 0.0, f"Insufficient speech. Only {voiced_percentage:.1f}% voiced. Please speak clearly."
            
            if len(audio_speech) < int(self.sample_rate * 0.3):
                return False, 0.0, "Speech too brief. Please speak for at least half a second."
            
            print(f" Extracted {len(audio_speech)/self.sample_rate:.2f}s of speech\n")
            
            # STEP 5: Extract MFCC and generate embedding
            print("STEP 5/6: Extracting features and generating embedding...")
            mfcc = self.mfcc_extractor.extract_for_cnn(audio_speech, target_frames=100)
            live_embedding = self.embedding_extractor.extract_embedding(mfcc)
            
            print(f" Live embedding generated (norm={np.linalg.norm(live_embedding):.6f})\n")
            
            # STEP 6: Compare embeddings
            print("STEP 6/6: Comparing voice embeddings...")
            similarity = self._compute_similarity(live_embedding, stored_embedding)
            
            print(f" Cosine similarity: {similarity:.4f}")
            print(f" Threshold: {self.threshold:.4f}")
            
            # Make decision
            is_authenticated = similarity >= self.threshold
            
            print(f"\n{'='*60}")
            if is_authenticated:
                print(f" AUTHENTICATED - User '{user_id}' verified")
                print(f"   Similarity: {similarity:.4f} ({similarity:.2%})")
                print(f"   Status: MATCH - Same speaker detected")
                message = f"Authentication successful (similarity: {similarity:.2%})"
            else:
                print(f" REJECTED - Different speaker detected")
                print(f"   Similarity: {similarity:.4f} ({similarity:.2%})")
                print(f"   Required: {self.threshold:.4f} ({self.threshold:.2%})")
                print(f"   Status: NO MATCH - Voice does not match enrolled user")
                message = f"Authentication failed. Different speaker detected (similarity: {similarity:.2%}, required: {self.threshold:.2%})"
            print(f"{'='*60}\n")
            
            return is_authenticated, similarity, message
            
        except Exception as e:
            error_msg = f"Authentication failed: {str(e)}"
            print(f"\n {error_msg}\n")
            return False, 0.0, error_msg
    
    def _validate_human_voice(self, audio: np.ndarray) -> Tuple[bool, str]:
        """
        Validate human voice (same as enrollment).
        """
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        total_energy = np.sum(magnitude ** 2)
        if total_energy < 1e-10:
            return False, "No audio signal detected"
        
        voice_mask = (freqs >= 80) & (freqs <= 3400)
        voice_energy = np.sum(magnitude[voice_mask] ** 2)
        voice_ratio = voice_energy / total_energy
        
        print(f" Voice frequency energy: {voice_ratio:.2%}")
        
        if voice_ratio < 0.10:
            return False, f"No human voice detected (voice energy: {voice_ratio:.2%}). Please speak clearly."
        
        return True, ""
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2
        
        # Compute dot product (cosine similarity for unit vectors)
        similarity = np.dot(embedding1_normalized, embedding2_normalized)
        
        # Clip to valid range (handle numerical errors)
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return float(similarity)
    
    def close(self):
        """Close resources."""
        self.voice_capture.close()


def main():
    """Test authentication pipeline."""
    print("\n" + "="*60)
    print("VOICE AUTHENTICATION - LOGIN TEST")
    print("="*60 + "\n")
    
    # Initialize pipeline
    pipeline = AuthenticationPipeline(threshold=0.80)
    
    try:
        print("This test requires a previously enrolled user.")
        print("For a real test, first run enrollment_pipeline.py\n")
        
        # For demonstration, create a random stored embedding
        stored_embedding = np.random.randn(128)
        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
        
        print(f"Using dummy stored embedding (norm={np.linalg.norm(stored_embedding):.6f})\n")
        
        # Test authentication
        user_id = "test_user_001"
        is_authenticated, similarity, message = pipeline.authenticate_user(user_id, stored_embedding)
        
        if is_authenticated:
            print(f"\n Authentication successful!")
            print(f"   User: {user_id}")
            print(f"   Similarity: {similarity:.4f} ({similarity:.2%})")
        else:
            print(f"\n Authentication failed!")
            print(f"   Reason: {message}")
            print(f"   Similarity: {similarity:.4f} ({similarity:.2%})")
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
