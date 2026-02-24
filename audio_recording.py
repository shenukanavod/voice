"""
Audio Recording Module
Handles real-time voice recording from microphone
"""

import pyaudio
import numpy as np
from typing import Tuple, Optional
from app.config import settings


class VoiceRecorder:
    """Record voice samples from microphone."""

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.device_index = device_index
        self.chunk_size = 1024

    def record(self) -> np.ndarray:
        """
        Record audio from microphone.

        Returns:
            Audio array (float32, normalized to [-1, 1])
        """
        pyaudio_instance = pyaudio.PyAudio()

        try:
            stream = pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )

            frames = []
            num_chunks = int(self.sample_rate / self.chunk_size * self.duration)

            print(f"🎤 Recording for {self.duration}s...")
            for _ in range(num_chunks):
                data = stream.read(self.chunk_size)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # Convert to numpy array
            audio_bytes = b"".join(frames)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

            print(f"✅ Recording complete: {len(audio_array)} samples")
            return audio_array

        finally:
            pyaudio_instance.terminate()

    def list_devices(self) -> list:
        """List available audio input devices."""
        pyaudio_instance = pyaudio.PyAudio()
        devices = []

        for i in range(pyaudio_instance.get_device_count()):
            try:
                info = pyaudio_instance.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices.append(
                        {
                            "index": i,
                            "name": info["name"],
                            "channels": info["maxInputChannels"],
                            "sample_rate": int(info["defaultSampleRate"]),
                        }
                    )
            except:
                pass

        pyaudio_instance.terminate()
        return devices

    def test_microphone(self) -> bool:
        """Test if microphone is working."""
        try:
            pyaudio_instance = pyaudio.PyAudio()
            stream = pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )
            stream.close()
            pyaudio_instance.terminate()
            print("✓ Microphone test successful")
            return True
        except Exception as e:
            print(f"✗ Microphone test failed: {e}")
            return False
