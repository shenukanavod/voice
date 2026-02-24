"""
VAD Processor Module - Voice Activity Detection using WebRTC-VAD.
Detects voiced segments and merges them to avoid fragmentation.
"""

import numpy as np
import webrtcvad
from typing import List, Tuple


class VADProcessor:
    """
    Voice Activity Detection processor using WebRTC-VAD.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 aggressiveness: int = 2,
                 frame_duration_ms: int = 30):
    
        # Validate sample rate
        valid_rates = [8000, 16000, 32000, 48000]
        if sample_rate not in valid_rates:
            raise ValueError(f"Sample rate must be one of {valid_rates}")
        
        # Validate frame duration
        valid_durations = [10, 20, 30]
        if frame_duration_ms not in valid_durations:
            raise ValueError(f"Frame duration must be one of {valid_durations}")
        
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Merging parameters
        self.max_gap_ms = 300  # Maximum gap to bridge (ms)
        self.min_segment_ms = 250  # Minimum segment duration (ms)
    
    def detect_voice(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect voice activity in audio.
        
        """
        print("Running Voice Activity Detection...")
        
        # Convert to int16 for WebRTC-VAD
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Detect voiced frames
        voiced_frames = self._detect_frames(audio_int16)
        
        # Calculate voiced percentage
        total_frames = len(voiced_frames)
        voiced_count = sum(voiced_frames)
        voiced_percentage = (voiced_count / total_frames * 100) if total_frames > 0 else 0
        
        print(f" Voiced frames: {voiced_percentage:.1f}%")
        
        # If no voiced frames, return empty
        if voiced_count == 0:
            print(" No voiced frames detected")
            return np.array([]), 0.0
        
        # Merge segments to avoid fragmentation
        merged_segments = self._merge_segments(voiced_frames)
        
        # Extract voiced audio
        voiced_audio = self._extract_segments(audio, merged_segments)
        
        print(f" VAD complete - extracted {len(voiced_audio)/self.sample_rate:.2f}s of speech")
        
        return voiced_audio, voiced_percentage
    
    def _detect_frames(self, audio_int16: np.ndarray) -> List[bool]:
        """
        Detect voiced frames using WebRTC-VAD.
      
        """
        voiced_frames = []
        
        # Process audio in frames
        num_frames = len(audio_int16) // self.frame_size
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio_int16[start:end]
            
            # Check if frame is too short
            if len(frame) < self.frame_size:
                break
            
            # Convert to bytes
            frame_bytes = frame.tobytes()
            
            # Check if frame contains voice
            try:
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                voiced_frames.append(is_speech)
            except Exception as e:
                # If VAD fails, assume not speech
                voiced_frames.append(False)
        
        return voiced_frames
    
    def _merge_segments(self, voiced_frames: List[bool]) -> List[Tuple[int, int]]:
        """
        Merge close voiced segments to avoid fragmentation.
        
        """
        segments = []
        max_gap_frames = int(self.max_gap_ms / self.frame_duration_ms)
        min_segment_frames = int(self.min_segment_ms / self.frame_duration_ms)
        
        # Find all voiced segments
        current_segment_start = None
        last_voiced_frame = None
        
        for i, is_voiced in enumerate(voiced_frames):
            if is_voiced:
                if current_segment_start is None:
                    # Start new segment
                    current_segment_start = i
                last_voiced_frame = i
            else:
                if current_segment_start is not None and last_voiced_frame is not None:
                    # Check if gap is too large
                    gap = i - last_voiced_frame
                    if gap > max_gap_frames:
                        # End current segment
                        segment_length = last_voiced_frame - current_segment_start + 1
                        if segment_length >= min_segment_frames:
                            segments.append((current_segment_start, last_voiced_frame + 1))
                        current_segment_start = None
                        last_voiced_frame = None
        
        # Add final segment if exists
        if current_segment_start is not None and last_voiced_frame is not None:
            segment_length = last_voiced_frame - current_segment_start + 1
            if segment_length >= min_segment_frames:
                segments.append((current_segment_start, last_voiced_frame + 1))
        
        print(f" Found {len(segments)} voiced segments after merging")
        
        return segments
    
    def _extract_segments(self, audio: np.ndarray, segments: List[Tuple[int, int]]) -> np.ndarray:
        """
        Extract audio from specified segments.
        
        
        """
        if not segments:
            return np.array([])
        
        extracted = []
        
        for start_frame, end_frame in segments:
            # Convert frame indices to sample indices
            start_sample = start_frame * self.frame_size
            end_sample = end_frame * self.frame_size
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            extracted.append(segment_audio)
        
        # Concatenate all segments
        return np.concatenate(extracted)
    
    def validate_speech(self, audio: np.ndarray, 
                       min_voiced_percentage: float = 30.0,
                       min_duration_seconds: float = 0.3) -> Tuple[bool, str]:
        """
        Validate if audio contains sufficient speech.
        
        """
        # Run VAD
        voiced_audio, voiced_percentage = self.detect_voice(audio)
        
        # Check voiced percentage
        if voiced_percentage < min_voiced_percentage:
            return False, f"Insufficient speech detected. Only {voiced_percentage:.1f}% voiced frames (minimum: {min_voiced_percentage:.0f}%). Please speak clearly."
        
        # Check speech duration
        speech_duration = len(voiced_audio) / self.sample_rate
        if speech_duration < min_duration_seconds:
            return False, f"Speech too brief ({speech_duration:.2f}s). Please speak for at least {min_duration_seconds}s."
        
        return True, ""


def test_vad():
    """Test VAD processor."""
    print("=== VAD Processor Test ===")
    
    # Generate test audio: speech-like signal with pauses
    sample_rate = 16000
    duration = 3.0
    
    # Create audio with speech and silence
    audio = []
    
    # Speech segment 1 (0.0 - 0.8s)
    t1 = np.linspace(0, 0.8, int(sample_rate * 0.8))
    speech1 = 0.3 * np.sin(2 * np.pi * 200 * t1)  # 200 Hz
    audio.extend(speech1)
    
    # Silence (0.8 - 1.2s)
    silence = np.zeros(int(sample_rate * 0.4))
    audio.extend(silence)
    
    # Speech segment 2 (1.2 - 2.0s)
    t2 = np.linspace(0, 0.8, int(sample_rate * 0.8))
    speech2 = 0.3 * np.sin(2 * np.pi * 250 * t2)  # 250 Hz
    audio.extend(speech2)
    
    # Silence (2.0 - 3.0s)
    silence2 = np.zeros(int(sample_rate * 1.0))
    audio.extend(silence2)
    
    audio = np.array(audio).astype(np.float32)
    
    # Test VAD
    vad = VADProcessor(sample_rate=sample_rate)
    voiced_audio, voiced_percentage = vad.detect_voice(audio)
    
    print(f"Original duration: {len(audio)/sample_rate:.2f}s")
    print(f"Voiced duration: {len(voiced_audio)/sample_rate:.2f}s")
    print(f"Voiced percentage: {voiced_percentage:.1f}%")
    
    # Validate speech
    is_valid, error = vad.validate_speech(audio)
    if is_valid:
        print(" Speech validation passed")
    else:
        print(f" Speech validation failed: {error}")


if __name__ == "__main__":
    test_vad()
