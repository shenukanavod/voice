"""
Audio Processing for Voice Authentication
Handles recording, preprocessing, and feature extraction
"""

import numpy as np
import librosa
import pyaudio
import wave
from pathlib import Path
from typing import Tuple, Optional
import io


class AudioProcessor:
    """Handles audio recording and preprocessing."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 duration: float = 3.0,
                 n_mels: int = 120):
     
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.chunk_size = 1024
        
        print(f"✅ Audio processor initialized")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Duration: {duration}s")
        print(f"   Mel bands: {n_mels}")
    
    def record_audio(self, verbose: bool = True) -> np.ndarray:
       
        if verbose:
            print(f"\n🎤 Recording for {self.duration} seconds...")
            print("   Speak clearly into your microphone...")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        try:
            # Open stream
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            num_chunks = int(self.sample_rate / self.chunk_size * self.duration)
            
            # Record
            for _ in range(num_chunks):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            if verbose:
                print("✅ Recording complete")
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            return audio_array
            
        finally:
            audio.terminate()
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio from file.
       
        """
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        # Trim or pad to desired duration
        target_length = int(self.sample_rate * self.duration)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        return audio
    
    def save_audio(self, audio: np.ndarray, file_path: str):
        """
        Save audio to WAV file.
        

        """
        # Convert to int16
        audio_int = (audio * 32767).astype(np.int16)
        
        # Save as WAV
        with wave.open(file_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int.tobytes())
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.
        
        """
        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512,
            fmin=20,
            fmax=8000
        )
        
        # Convert to log scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        
        return mel_db
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio: validate and extract features.
      
        """
        # Check if audio has sufficient energy
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01:
            raise ValueError("Audio is too quiet. Please speak louder.")
        
        # Extract mel spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        return mel_spec
    
    def validate_audio(self, audio: np.ndarray) -> Tuple[bool, str]:
        """
        Validate if audio contains speech.
        
        """
        # Check RMS energy
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01:
            return False, "Audio is too quiet"
        
        # Check if audio is not silent
        if np.max(np.abs(audio)) < 0.01:
            return False, "No audio detected"
        
        # Check for clipping
        if np.max(np.abs(audio)) > 0.95:
            return False, "Audio is clipping (too loud)"
        
        return True, "Audio is valid"
