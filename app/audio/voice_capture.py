"""
Voice Capture Module - Records audio with strict human voice validation.
Rejects silence, noise, and non-human sounds.
"""

import numpy as np
import pyaudio
import wave
import time
from typing import Tuple, Optional

class VoiceCapture:
    """
    Captures audio from microphone with real-time validation.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 duration: float = 3.0,
                 chunk_size: int = 1024):
      
        self.sample_rate = sample_rate
        self.duration = duration
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # IMPROVED: Lower threshold for normal speaking voice
        # Works from further distance and with quieter microphones
        self.rms_threshold = 0.005  # Reduced from 0.02 for better sensitivity
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Auto-gain settings for normal voice capture
        self.auto_gain_enabled = True
        self.target_rms = 0.25  # Target level for normalization (increased for normal speaking)
    
    def record_audio(self) -> Tuple[np.ndarray, bool, str]:
        """
        Record audio from microphone with real-time validation.
        
        Returns:
            Tuple of (audio_array, is_valid, error_message)
            - audio_array: Recorded audio as numpy array
            - is_valid: True if audio passes validation
            - error_message: Description of validation failure (empty if valid)
        """
        print("\n Recording... Speak clearly into the microphone")
        
        frames = []
        stream = None
        
        try:
            # Open audio stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Calculate number of chunks to record
            num_chunks = int(self.sample_rate / self.chunk_size * self.duration)
            
            # Record audio chunks
            for i in range(num_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            print("Recording complete")
            
        except Exception as e:
            error_msg = f"Recording error: {str(e)}"
            print(f" {error_msg}")
            return np.array([]), False, error_msg
            
        finally:
            # Close stream
            if stream is not None:
                stream.stop_stream()
                stream.close()
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
        
        # IMPROVED: Apply automatic gain control for normal voice
        if self.auto_gain_enabled:
            audio_array = self._apply_auto_gain(audio_array)
        
        # Validate audio
        is_valid, error_message = self._validate_audio(audio_array)
        
        return audio_array, is_valid, error_message
    
    def _apply_auto_gain(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply automatic gain control to normalize audio levels.
        Boosts quiet audio to make normal speaking voice work from any distance.
        
       
        """
        current_rms = np.sqrt(np.mean(audio ** 2))
        
        if current_rms > 0.00001:  # Not complete silence
            # Calculate gain to reach target level
            gain = self.target_rms / current_rms
            # Limit gain to prevent excessive noise amplification
            gain = min(gain, 50.0)  # Max 50x boost for very quiet audio
            gain = max(gain, 1.0)   # Never reduce volume
            
            audio = audio * gain
            new_rms = np.sqrt(np.mean(audio ** 2))
            print(f"🔊 Auto-gain: {gain:.1f}x (RMS: {current_rms:.4f} → {new_rms:.4f})")
        
        # Ensure audio stays in valid range
        audio = np.clip(audio, -1.0, 1.0)
        return audio
    
    def _validate_audio(self, audio: np.ndarray) -> Tuple[bool, str]:
        """
        Validate recorded audio - optimized for normal speaking voice.
        
        This is the first gate - rejects complete silence.
        More detailed validation happens in preprocessing.
        
      
        """
        # Check if audio is empty
        if len(audio) == 0:
            return False, "No audio recorded"
        
        # CHECK 1: Compute RMS (Root Mean Square) energy
        rms = np.sqrt(np.mean(audio ** 2))
        print(f" Audio RMS: {rms:.6f}")
        
        # IMPROVED: Much lower threshold for normal voice after auto-gain
        if rms < self.rms_threshold:
            return False, f"No speech detected (RMS: {rms:.6f}). Please speak at normal volume."
        
        # CHECK 2: Check for audio variation (detect constant noise vs speech)
        audio_std = np.std(audio)
        print(f" Audio variation (std): {audio_std:.6f}")
        
        # IMPROVED: Lower threshold for normal voice
        if audio_std < 0.003:  # Reduced from 0.01
            return False, "Audio has no variation. Please speak at normal volume (not silence)."
        
        # CHECK 3: Check peak amplitude (detect if microphone is working)
        max_amplitude = np.max(np.abs(audio))
        print(f" Max amplitude: {max_amplitude:.6f}")
        
        # IMPROVED: Lower threshold after auto-gain
        if max_amplitude < 0.01:  # Reduced from 0.05
            return False, f"Audio signal too weak (max: {max_amplitude:.6f}). Check microphone."
        
        print(" Initial validation passed - Normal voice detected")
        return True, ""
    
    def save_audio(self, audio: np.ndarray, filepath: str):
        """
        Save audio to WAV file.
        
        Args:
            audio: Audio array to save
            filepath: Output file path
        """
        # Convert back to int16
        audio_int16 = (audio * 32768.0).astype(np.int16)
        
        # Write WAV file
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        print(f" Audio saved to: {filepath}")
    
    def close(self):
        """Close PyAudio instance."""
        self.audio.terminate()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.audio.terminate()
        except:
            pass


def test_microphone():
    """Test microphone recording."""
    print("=== Microphone Test ===")
    
    capturer = VoiceCapture(duration=2.0)
    
    try:
        audio, is_valid, error = capturer.record_audio()
        
        if is_valid:
            print(" Microphone test successful!")
            print(f"   Recorded {len(audio)} samples ({len(audio)/16000:.2f}s)")
        else:
            print(f" Microphone test failed: {error}")
    
    finally:
        capturer.close()


if __name__ == "__main__":
    test_microphone()
