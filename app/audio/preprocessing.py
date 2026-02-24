import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, List
import warnings
from scipy import signal
from sklearn.preprocessing import StandardScaler

# Optional import for webrtcvad
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    webrtcvad = None

from app.config import settings

warnings.filterwarnings("ignore", category=UserWarning)

class AudioPreprocessor:
    """
    Audio preprocessing pipeline for voice authentication.
    
    Features:
    - Voice Activity Detection (VAD)
    - Noise reduction
    - Normalization
    - Feature extraction (MFCC, Spectrogram, Mel-spectrogram)
    """
    
    def __init__(self):
        self.sample_rate = settings.SAMPLE_RATE
        self.duration = settings.AUDIO_DURATION
        self.n_mfcc = settings.N_MFCC
        self.n_mels = settings.N_MELS
        self.hop_length = settings.HOP_LENGTH
        self.n_fft = settings.N_FFT
        
        # Initialize VAD (optional)
        if WEBRTCVAD_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        else:
            self.vad = None
        
        # Initialize scalers for feature normalization
        self.mfcc_scaler = StandardScaler()
        self.spectrogram_scaler = StandardScaler()
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {str(e)}")
    
    def preprocess_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """
        Preprocess audio from bytes (for real-time processing).
       
        """
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize to [-1, 1]
            
            return self.preprocess_audio(audio)
        except Exception as e:
            raise ValueError(f"Error preprocessing audio bytes: {str(e)}")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Complete audio preprocessing pipeline with STRICT speech validation.
        Rejects silence, background noise, and insufficient speech.

        """
        # CRITICAL: Validate input
        if audio is None:
            raise ValueError("Audio input is None! Cannot preprocess.")
        
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"Audio must be numpy array, got {type(audio)}")
        
        if len(audio) == 0:
            raise ValueError("Audio array is empty! Cannot preprocess.")
        
        print(f"\n{'='*60}")
        print("STRICT VOICE VALIDATION STARTING")
        print(f"{'='*60}")
        print(f"Raw audio length: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
        
        # IMPROVED AUTO GAIN CONTROL: Better handling of normal voice from any distance
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0.000001:  # Not complete silence (lower threshold)
            # Boost audio to higher target for better feature extraction
            target_rms = 0.35  # Increased from 0.15 to allow normal speaking volume
            gain = target_rms / current_rms
            # More aggressive gain for distant/quiet speakers
            gain = min(gain, 200.0)  # Increased from 100x to 200x max boost
            gain = max(gain, 0.5)     # Allow slight reduction if too loud
            audio = audio * gain
            print(f"Auto-gain applied: {gain:.1f}x boost (RMS: {current_rms:.6f} → {np.sqrt(np.mean(audio**2)):.6f})")
        
        # STRICT CHECK 1: Compute RMS - REJECT SILENCE IMMEDIATELY
        rms = np.sqrt(np.mean(audio ** 2))
        print(f"\nCHECK 1: RMS Energy")
        print(f"   RMS: {rms:.6f}")
        
        # IMPROVED: Much lower threshold for normal voice after auto-gain
        if rms < 0.00005:  # Further reduced for normal speaking voice
            print(f"   REJECTED: Complete silence (RMS < 0.00005)")
            raise ValueError(f"NO SPEECH DETECTED — Please speak at normal volume.\n\nYour audio volume (RMS: {rms:.6f}) is too low.\nSpeak naturally - no need to shout!")
        print(f"    PASSED: Normal voice energy detected")
        
        # STRICT CHECK 2: Audio variation (detect constant noise vs real speech)
        audio_std = np.std(audio)
        print(f"\n CHECK 2: Audio Variation")
        print(f"   Std Dev: {audio_std:.6f}")
        
        if audio_std < 0.00005:  # Further lowered for normal voice
            print(f"   REJECTED: No variation (constant noise)")
            raise ValueError(f"No human voice detected. Audio is too uniform (constant noise/silence). Speak at normal volume.")
        
        print(f"   PASSED: Audio has variation")
        
        # STRICT CHECK 3: Detect silence segments
        segment_duration = 0.1  # 100ms segments
        segment_samples = int(self.sample_rate * segment_duration)
        num_segments = len(audio) // segment_samples
        
        silent_segments = 0
        voiced_segments = 0
        
        print(f"\n CHECK 3: Segment Analysis ({num_segments} segments)")
        
        for i in range(num_segments):
            segment = audio[i * segment_samples:(i + 1) * segment_samples]
            segment_rms = np.sqrt(np.mean(segment ** 2))
            
            if segment_rms < 0.00005:  # Further adjusted for normal voice
                silent_segments += 1
            else:
                voiced_segments += 1
        
        silence_percentage = (silent_segments / num_segments) * 100 if num_segments > 0 else 100
        voiced_percentage = (voiced_segments / num_segments) * 100 if num_segments > 0 else 0
        
        print(f"   Silent segments: {silence_percentage:.1f}%")
        print(f"   Voiced segments: {voiced_percentage:.1f}%")
        
        # CRITICAL: Reject if >70% silence (relaxed for quiet microphones)
        if silence_percentage > 70:  # More lenient threshold
            print(f"    REJECTED: Too much silence ({silence_percentage:.1f}%)")
            raise ValueError(f" TOO MUCH SILENCE — Audio contains {silence_percentage:.1f}% silence.\n\nPlease speak continuously for the full recording duration.\n(Maximum 70% silence allowed)")
        
        print(f"   PASSED: Sufficient voiced segments ({voiced_percentage:.1f}%)")
        
        # STRICT CHECK 4: Verify continuous speech (must speak for 0.3s minimum)
        print(f"\n CHECK 4: Continuous Speech Detection")
        continuous_speech = self._check_continuous_speech(audio)
        print(f"   Continuous speech: {continuous_speech:.2f}s")
        
        if continuous_speech < 0.3:  # More relaxed for various microphones
            print(f"    REJECTED: Speech too short ({continuous_speech:.2f}s < 0.3s)")
            raise ValueError(f" SPEECH TOO SHORT — Detected only {continuous_speech:.2f}s of continuous speech.\n\nPlease speak for at least 0.3 seconds.\n(Example: Say 'Hello, testing')")
        
        print(f"   PASSED: Sufficient continuous speech")
        
        # Ensure audio is not clipped
        audio = np.clip(audio, -1.0, 1.0)
        
        # STRICT CHECK 5: Human voice frequency detection (before VAD)
        print(f"\n CHECK 5: Human Speech Frequency Analysis")
        speech_energy = self._check_speech_frequencies(audio)
        print(f"   Speech frequency energy: {speech_energy:.3f}")
        
        # IMPROVED: Lower threshold for normal voice from any distance
        if speech_energy < 0.02:  # Further lowered from 0.05 for normal voice
            print(f"    REJECTED: No human voice frequencies ({speech_energy:.3f} < 0.02)")
            raise ValueError(f" NO HUMAN VOICE DETECTED — Audio does not contain human speech frequencies.\n\nDetected energy: {speech_energy:.3f} (need ≥0.02)\nPlease speak clearly into the microphone.\n(Background noise, music, or non-speech sounds rejected)")
        
        print(f"    PASSED: Human voice frequencies detected")
        
        # Step 2: Ensure audio is the right length (pad or truncate)
        audio = self._pad_or_truncate(audio)
        
        # STRICT CHECK 6: Apply VAD with strict validation (40% voiced minimum)
        print(f"\n CHECK 6: Voice Activity Detection (VAD)")
        audio_after_vad = self._apply_vad_strict(audio)
        
        print(f"   After VAD: {len(audio_after_vad)} samples ({len(audio_after_vad)/self.sample_rate:.2f}s)")
        
        # STRICT CHECK 7: Minimum speech duration (0.5s after VAD for normal speaking)
        print(f"\n CHECK 7: Post-VAD Duration Validation")
        min_speech_samples = int(self.sample_rate * 0.5)
        actual_duration = len(audio_after_vad) / self.sample_rate
        print(f"   Duration after VAD: {actual_duration:.2f}s")
        
        if len(audio_after_vad) < min_speech_samples:
            print(f"    REJECTED: Too short after VAD ({actual_duration:.2f}s < 0.5s)")
            raise ValueError(f" Insufficient speech detected. Only {actual_duration:.2f}s of speech found (minimum: 0.5s). Please speak clearly and continuously.")
        
        print(f"    PASSED: Sufficient duration after VAD")
        
        # Step 4: Trim only leading/trailing silence
        audio_trimmed = self._trim_silence(audio_after_vad)
        
        print(f"\n After trim: {len(audio_trimmed)} samples ({len(audio_trimmed)/self.sample_rate:.2f}s)")
        
        # STRICT CHECK 8: Final validation after trimming
        trimmed_duration = len(audio_trimmed) / self.sample_rate
        if len(audio_trimmed) < min_speech_samples:
            print(f"    REJECTED: Too short after trimming ({trimmed_duration:.2f}s < 0.5s)")
            raise ValueError(f" Insufficient speech after trimming. Only {trimmed_duration:.2f}s remaining (minimum: 0.5s). Please speak more.")
        
        # Step 5: Normalize audio
        audio_normalized = self._normalize_audio(audio_trimmed)
        
        # Step 6: Ensure float32 dtype
        audio_normalized = audio_normalized.astype(np.float32)
        
        # FINAL SUCCESS REPORT
        final_rms = np.sqrt(np.mean(audio_normalized ** 2))
        final_duration = len(audio_normalized) / self.sample_rate
        
        print(f"\n{'='*60}")
        print(f" ALL VALIDATION CHECKS PASSED!")
        print(f"{'='*60}")
        print(f"Final audio stats:")
        print(f"  • Duration: {final_duration:.2f}s")
        print(f"  • Samples: {len(audio_normalized)}")
        print(f"  • RMS Energy: {final_rms:.6f}")
        print(f"  • Format: {audio_normalized.dtype}")
        print(f"{'='*60}\n")
        
        return audio_normalized
    
    def _pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to target duration."""
        target_length = int(self.sample_rate * self.duration)
        
        if len(audio) > target_length:
            # Truncate from center
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
    
    def _apply_vad(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Voice Activity Detection to remove silence.
        Uses energy-based VAD with aggressiveness = 0 (least aggressive).
        """
        # If webrtcvad is not available, use energy-based VAD
        if not WEBRTCVAD_AVAILABLE or self.vad is None:
            return self._energy_based_vad(audio)
            
        try:
            # Convert to 16-bit PCM for VAD
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Frame size for VAD (10, 20, or 30 ms) - use 10ms for less aggressive
            frame_duration = 10  # ms (shorter = less aggressive)
            frame_size = int(self.sample_rate * frame_duration / 1000)
            
            # Recreate VAD with aggressiveness 0 (least aggressive)
            vad = webrtcvad.Vad(0)  # 0 = least aggressive
            
            # Process audio in frames
            voiced_frames = []
            for i in range(0, len(audio_int16) - frame_size + 1, frame_size):
                frame = audio_int16[i:i + frame_size]
                frame_bytes = frame.tobytes()
                
                # Check if frame contains speech
                if vad.is_speech(frame_bytes, self.sample_rate):
                    voiced_frames.extend(frame)
            
            if voiced_frames:
                return np.array(voiced_frames, dtype=np.float32) / 32767.0
            else:
                # If no speech detected, return original audio
                print(" WebRTC VAD found no speech, using original audio")
                return audio
                
        except Exception as e:
            print(f" WebRTC VAD error: {e}, falling back to energy-based VAD")
            # If VAD fails, use energy-based fallback
            return self._energy_based_vad(audio)
    
    def _energy_based_vad(self, audio: np.ndarray) -> np.ndarray:
        """
        Energy-based VAD fallback with lenient threshold.
        Keeps short pauses and doesn't fragment speech.
        """
        try:
            # Frame settings (20ms frames for smoother detection)
            frame_duration = 20  # milliseconds (longer frames = less fragmentation)
            frame_size = int(self.sample_rate * frame_duration / 1000)
            
            # Energy threshold (very lenient to preserve speech)
            energy_threshold = 0.005  # Lowered from 0.01
            
            # Calculate energy for all frames
            frame_energies = []
            for i in range(0, len(audio) - frame_size, frame_size):
                frame = audio[i:i + frame_size]
                energy = np.sqrt(np.mean(frame ** 2))
                frame_energies.append(energy)
            
            # Keep frames above threshold OR between speech frames (preserve pauses)
            voiced_indices = []
            for i, energy in enumerate(frame_energies):
                if energy > energy_threshold:
                    voiced_indices.append(i)
            
            if len(voiced_indices) > 0:
                # Include frames between first and last voiced frame (keeps pauses)
                start_idx = voiced_indices[0] * frame_size
                end_idx = (voiced_indices[-1] + 1) * frame_size
                return audio[start_idx:end_idx].astype(np.float32)
            else:
                # If no speech detected, return original audio
                return audio
                
        except Exception as e:
            print(f" Energy VAD error: {e}, returning original audio")
            return audio
    
    def _apply_vad_strict(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply VAD with STRICT validation - requires 40% voiced frames minimum.
        
        """
        original_length = len(audio)
        
        # Apply VAD
        vad_audio = self._apply_vad(audio)
        
        # Calculate voiced percentage
        voiced_percentage = (len(vad_audio) / original_length) * 100 if original_length > 0 else 0
        print(f"   Voiced frames after VAD: {voiced_percentage:.1f}%")
        
        # STRICT: Reject if <40% voiced (increased from 20%)
        if voiced_percentage < 40:
            print(f"    REJECTED: Insufficient speech ({voiced_percentage:.1f}% < 40%)")
            raise ValueError(f" Insufficient speech detected. Only {voiced_percentage:.1f}% voiced frames (minimum: 40%). Please speak clearly and continuously.")
        
        # STRICT: Reject if >60% removed (increased from 80%)
        if len(vad_audio) < (original_length * 0.4):
            removed_percentage = 100 * (1 - len(vad_audio) / original_length)
            print(f"    REJECTED: Too much silence removed ({removed_percentage:.1f}%)")
            raise ValueError(f" No valid speech detected. VAD removed {removed_percentage:.1f}% of audio. Please speak clearly without long pauses.")
        
        print(f"    PASSED: Sufficient voiced content")
        return vad_audio
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Trim only leading and trailing silence, keep internal pauses.
        
        """
        try:
            # Calculate frame energy
            frame_length = int(self.sample_rate * 0.02)  # 20ms frames
            
            # Find first non-silent frame
            start_idx = 0
            for i in range(0, len(audio) - frame_length, frame_length):
                frame = audio[i:i + frame_length]
                if np.sqrt(np.mean(frame ** 2)) > threshold:
                    start_idx = max(0, i - frame_length)  # Include one frame before
                    break
            
            # Find last non-silent frame
            end_idx = len(audio)
            for i in range(len(audio) - frame_length, 0, -frame_length):
                frame = audio[i:i + frame_length]
                if np.sqrt(np.mean(frame ** 2)) > threshold:
                    end_idx = min(len(audio), i + 2 * frame_length)  # Include one frame after
                    break
            
            # Return trimmed audio
            if start_idx < end_idx:
                return audio[start_idx:end_idx]
            else:
                return audio
                
        except Exception as e:
            print(f" Trim error: {e}, returning original audio")
            return audio
    
    def _check_speech_frequencies(self, audio: np.ndarray) -> float:
        """
        Check if audio contains human speech frequencies.
        Human speech has energy concentrated in:
        - Fundamental: 85-255 Hz
        - Formants: 500-8000 Hz
        
        Returns:
            Speech energy ratio (0-1), higher = more likely to be speech
        """
        try:
            # Compute FFT
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            
            # Frequency bins
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            
            # Total energy
            total_energy = np.sum(magnitude ** 2)
            
            if total_energy < 1e-10:
                return 0.0
            
            # Speech frequency ranges
            # Fundamental frequency: 85-255 Hz (pitch range)
            fundamental_mask = (freqs >= 85) & (freqs <= 255)
            fundamental_energy = np.sum(magnitude[fundamental_mask] ** 2)
            
            # Formant frequencies: 500-8000 Hz (vowel/consonant characteristics)
            formant_mask = (freqs >= 500) & (freqs <= 8000)
            formant_energy = np.sum(magnitude[formant_mask] ** 2)
            
            # Combined speech energy
            speech_energy = fundamental_energy + formant_energy
            
            # Ratio of speech energy to total energy
            speech_ratio = speech_energy / total_energy
            
            return min(speech_ratio, 1.0)
            
        except Exception as e:
            print(f" Speech frequency check error: {e}")
            return 0.5  # Neutral score on error
    
    def _check_continuous_speech(self, audio: np.ndarray) -> float:
        """
        Check for continuous speech duration - STRICT validation.
        Requires 1.0s minimum continuous segment at higher energy threshold.
        
        """
        try:
            # Use energy-based detection with sliding window
            window_size = int(self.sample_rate * 0.1)  # 100ms windows
            hop_size = int(self.sample_rate * 0.05)  # 50ms hop
            
            # Calculate energy for each window
            energies = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                energy = np.sqrt(np.mean(window ** 2))
                energies.append(energy)
            
            energies = np.array(energies)
            
            # STRICT: Increased threshold from 0.03 to 0.05 (requires clearer speech)
            speech_threshold = 0.05
            
            # Find continuous speech segments
            is_speech = energies > speech_threshold
            
            # Find longest continuous True sequence
            max_duration = 0
            current_duration = 0
            
            for val in is_speech:
                if val:
                    current_duration += 1
                else:
                    max_duration = max(max_duration, current_duration)
                    current_duration = 0
            
            max_duration = max(max_duration, current_duration)
            
            # Convert to seconds (each window represents 50ms hop)
            duration_seconds = (max_duration * hop_size) / self.sample_rate
            
            return duration_seconds
            
        except Exception as e:
            print(f" Continuous speech check error: {e}")
            return 0.0  # Fail safe - reject on error
    
    def validate_speech(self, audio: np.ndarray) -> tuple[bool, str]:
       
        
        try:
            print("\n" + "="*60)
            print("STRICT SPEECH SECURITY VALIDATION")
            print("="*60)
            
            # ==========================================
            # SECURITY CHECK 1: RMS Energy - Must be loud enough
            # ==========================================
            rms = np.sqrt(np.mean(audio ** 2))
            print(f"\n CHECK 1: Audio Volume (RMS)")
            print(f"   Detected: {rms:.6f}")
            print(f"   Required: ≥0.018 (ADJUSTED)")
            
            if rms < 0.018:  # Adjusted from 0.025 to 0.018 for better sensitivity
                print(f"    REJECTED: Too quiet or silent")
                return False, f" NO SPEECH DETECTED\n\nPlease speak clearly and loudly into the microphone.\n(Volume too low: {rms:.4f} < 0.018)"
            print(f"   PASSED: Sufficient volume")
            
            # ==========================================
            # SECURITY CHECK 2: Voiced Content - Must be real speech, not noise
            # ==========================================
            original_length = len(audio)
            vad_audio = self._apply_vad(audio)
            voiced_percent = (len(vad_audio) / original_length) * 100 if original_length > 0 else 0
            
            print(f"\n CHECK 2: Voice Detection (VAD)")
            print(f"   Voiced content: {voiced_percent:.1f}%")
            print(f"   Required: ≥50% (STRICT)")
            
            if voiced_percent < 50:  # INCREASED from 20% - reject noise/silence
                print(f"    REJECTED: Too much silence/noise")
                return False, f" INSUFFICIENT SPEECH\n\nOnly {voiced_percent:.1f}% of audio contains voice.\nPlease speak continuously for at least 1 second.\n(Need ≥50% voiced content)"
            print(f"   PASSED: Sufficient voice content")
            
            # ==========================================
            # SECURITY CHECK 3: Speech Duration - Must speak long enough
            # ==========================================
            vad_duration = len(vad_audio) / self.sample_rate
            print(f"\n CHECK 3: Speech Duration")
            print(f"   Duration: {vad_duration:.2f} seconds")
            print(f"   Required: ≥1.0s (STRICT)")
            
            if vad_duration < 1.0:  # INCREASED from 0.3s - reject short noises/clicks
                print(f"    REJECTED: Speech too short")
                return False, f" SPEECH TOO SHORT\n\nDetected only {vad_duration:.2f} seconds of speech.\nPlease speak for at least 1 full second.\n(Example: Say a full sentence)"
            print(f"   PASSED: Sufficient duration")
            
            # ==========================================
            # SECURITY CHECK 4: Human Voice Frequencies - Must be actual voice
            # ==========================================
            speech_energy = self._check_speech_frequencies(audio)
            print(f"\n CHECK 4: Human Voice Frequency Analysis")
            print(f"   Speech energy: {speech_energy:.3f}")
            print(f"   Required: ≥0.20 (STRICT)")
            
            if speech_energy < 0.20:  # DOUBLED from 0.10 - reject non-human sounds
                print(f"    REJECTED: Not human voice")
                return False, f" NO HUMAN VOICE DETECTED\n\nDetected sound is not human speech (energy: {speech_energy:.3f}).\nPlease speak clearly into the microphone.\n(Background noise, music, or non-speech sounds rejected)"
            print(f"    PASSED: Human voice detected")
            
            # ==========================================
            # SECURITY CHECK 5: Continuous Speech - Must not be fragmented
            # ==========================================
            continuous_duration = self._check_continuous_speech(audio)
            print(f"\n CHECK 5: Continuous Speech Check")
            print(f"   Longest speech: {continuous_duration:.2f}s")
            print(f"   Required: ≥0.8s (STRICT)")
            
            if continuous_duration < 0.8:  # NEW check - reject choppy/fragmented audio
                print(f"    REJECTED: Speech too fragmented")
                return False, f" FRAGMENTED SPEECH"
            print(f"    PASSED: Continuous speech detected")
            
            # ==========================================
            # ALL SECURITY CHECKS PASSED
            # ==========================================
            print(f"\n{'='*60}")
            print(" ALL SECURITY CHECKS PASSED - SPEECH VALIDATED")
            print(f"{'='*60}\n")
            return True, ""
            
        except Exception as e:
            error_msg = f"Speech validation error: {str(e)}"
            print(f"\n VALIDATION ERROR: {error_msg}")
            return False, f" VALIDATION ERROR\n\n{error_msg}\n\nPlease try again."
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple noise reduction using spectral gating.
        """
        try:
            # Compute short-time Fourier transform
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor (bottom 10% of magnitude values)
            noise_floor = np.percentile(magnitude, 10)
            
            # Apply spectral gating
            mask = magnitude > (noise_floor * 2)
            magnitude_cleaned = magnitude * mask
            
            # Reconstruct audio
            stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
            audio_cleaned = librosa.istft(stft_cleaned, hop_length=self.hop_length)
            
            return audio_cleaned
            
        except Exception:
            # If noise reduction fails, return original audio
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
       
        # Step 1: Remove DC offset
        audio = audio - np.mean(audio)
        
        # Step 2: Peak normalization - scale to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Step 3: Scale to 95% to prevent clipping
        audio = audio * 0.95
        
        return audio
    
    def is_speech(self, audio: np.ndarray, threshold: float = 0.4) -> bool:
        """
        Detect if audio contains speech (not just noise or music).
        
        
        """
        try:
            # 1. Zero Crossing Rate (speech: 0.05-0.3, music: varies widely, noise: very high/low)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)
            zcr_score = 1.0 if 0.05 < zcr_mean < 0.3 else 0.0
            
            # 2. Spectral Centroid (speech: 1000-4000 Hz typical)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            centroid_mean = np.mean(spectral_centroids)
            centroid_score = 1.0 if 500 < centroid_mean < 5000 else 0.0
            
            # 3. Spectral Rolloff (speech has characteristic rolloff around 4-6 kHz)
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, roll_percent=0.85)[0]
            rolloff_mean = np.mean(rolloff)
            rolloff_score = 1.0 if 2000 < rolloff_mean < 8000 else 0.5
            
            # 4. Harmonic-Percussive Separation (speech is mostly harmonic)
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(audio)) + 1e-9)
            harmonic_score = min(harmonic_ratio * 1.5, 1.0)  # Speech should be >66% harmonic
            
            # 5. Energy distribution (speech has energy in voice range 80-8000 Hz)
            stft = np.abs(librosa.stft(audio))
            freqs = librosa.fft_frequencies(sr=self.sample_rate)
            voice_mask = (freqs >= 80) & (freqs <= 8000)
            voice_energy = np.sum(stft[voice_mask, :])
            total_energy = np.sum(stft) + 1e-9
            voice_ratio = voice_energy / total_energy
            energy_score = min(voice_ratio * 1.2, 1.0)
            
            # Weighted combination of scores
            speech_confidence = (
                zcr_score * 0.15 +
                centroid_score * 0.20 +
                rolloff_score * 0.15 +
                harmonic_score * 0.25 +
                energy_score * 0.25
            )
            
            is_speech = speech_confidence >= threshold
            
            return is_speech
            
        except Exception as e:
            # If detection fails, be conservative and allow it
            return True
        
        # Step 2: Normalize to [-1, 1] range (REMOVES volume as feature)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Step 3: Scale to 95% to leave headroom
        audio = audio * 0.95
        
        # Ensure no clipping
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Add delta and delta-delta features
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # Concatenate features
            features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
            
            return features
            
        except Exception as e:
            raise ValueError(f"Error extracting MFCC features: {str(e)}")
    
    def extract_improved_features(self, audio: np.ndarray, target_frames: int = 64) -> np.ndarray:
        """
        IMPROVED FEATURE EXTRACTION - Matches training pipeline.
        
        Extracts speaker-specific features that are volume-independent:
        - Mel-spectrogram (64 bands): Captures pitch and harmonic structure
        - MFCC (40 coefficients): Captures voice timbre and formants
        - Delta-MFCC (40 coefficients): Captures voice dynamics
        
        """
        try:
            # CRITICAL: Ensure minimum audio length
            min_samples = self.sample_rate * 2  # 2 seconds minimum
            if len(audio) < min_samples:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
            
            # Validate audio is not empty or all zeros
            if len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
                raise ValueError("Audio is empty or too quiet for feature extraction")
            
            # 1. Extract Mel-spectrogram (pitch and harmonic structure)
            # IMPORTANT: Use 64 mel bands to match improved training (not self.n_mels which is 128)
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=64,  # FIXED: 64 mel bands (matches training)
                n_fft=2048,
                hop_length=512,
                fmin=80,    # Focus on human voice range (80-8000 Hz)
                fmax=8000
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # 2. Extract MFCC (voice timbre and formants)
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=40,  # 40 MFCC coefficients
                n_fft=2048,
                hop_length=512
            )
            
            # 3. Extract Delta-MFCC (voice dynamics over time)
            # IMPORTANT: Only compute delta on MFCC (not on previous stacked features)
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # 4. Stack features: [Mel (64) + MFCC (40) + Delta-MFCC (40)] = 144 features
            features = np.vstack([mel_db, mfcc, mfcc_delta])
            
            # 5. CRITICAL: Ensure consistent time dimension
            # Pad or crop to fixed size (64 time frames = 2 seconds at hop_length=512)
            if features.shape[1] < target_frames:
                # Pad with zeros if too short
                pad_width = ((0, 0), (0, target_frames - features.shape[1]))
                features = np.pad(features, pad_width, mode='constant')
            elif features.shape[1] > target_frames:
                # Crop if too long
                features = features[:, :target_frames]
            
            # 6. Normalize features per-sample (removes absolute scale)
            # This is critical - makes features volume-independent
            mean = features.mean()
            std = features.std()
            if std > 1e-9:
                features = (features - mean) / std
            
            return features.astype(np.float32)
            
        except Exception as e:
            raise ValueError(f"Error extracting improved features: {str(e)}")
    
    def extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio.
        
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                fmax=self.sample_rate // 2
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return log_mel_spec
            
        except Exception as e:
            raise ValueError(f"Error extracting spectrogram: {str(e)}")
    
    def extract_combined_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract combined MFCC and spectrogram features.
       
        """
        # Extract MFCC features
        mfcc_features = self.extract_mfcc(audio)
        
        # Extract mel-spectrogram
        mel_spec = self.extract_spectrogram(audio)
        
        # Ensure both feature types have the same time dimension
        min_time_frames = min(mfcc_features.shape[1], mel_spec.shape[1])
        mfcc_features = mfcc_features[:, :min_time_frames]
        mel_spec = mel_spec[:, :min_time_frames]
        
        # Resize both to target height
        mfcc_resized = self._resize_feature_map(mfcc_features, target_height=32)
        mel_resized = self._resize_feature_map(mel_spec, target_height=32)
        
        # Concatenate features vertically (along height dimension)
        # Result shape: (64, time_frames) where 64 = 32 (MFCC) + 32 (Mel)
        combined = np.vstack([mfcc_resized, mel_resized])
        
        # Add channel dimension: (1, 64, time_frames)
        combined_features = np.expand_dims(combined, axis=0)
        
        return combined_features
    
    def _resize_feature_map(self, features: np.ndarray, target_height: int) -> np.ndarray:
        """
        Resize feature map to target height using interpolation.
        """
        from scipy.ndimage import zoom
        
        current_height, width = features.shape
        height_ratio = target_height / current_height
        
        # Use zoom to resize
        resized_features = zoom(features, (height_ratio, 1), order=1)
        
        return resized_features
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio array to file.
        
        """
        try:
            sf.write(output_path, audio, self.sample_rate)
        except Exception as e:
            raise ValueError(f"Error saving audio to {output_path}: {str(e)}")

class RealTimeAudioProcessor:
    """
    Real time audio processing for continuous monitoring.
    """
    
    def __init__(self, chunk_size: int = 1024):
        self.preprocessor = AudioPreprocessor()
        self.chunk_size = chunk_size
        self.audio_buffer = []
        self.buffer_duration = 5.0  # seconds
        self.max_buffer_size = int(settings.SAMPLE_RATE * self.buffer_duration)
    
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Add audio chunk to buffer and return processed audio if buffer is full.
        
        """
        # Add chunk to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Check if buffer is full
        if len(self.audio_buffer) >= self.max_buffer_size:
            # Extract audio segment
            audio_segment = np.array(self.audio_buffer[:self.max_buffer_size])
            
            # Remove processed audio from buffer (keep overlap)
            overlap_size = self.max_buffer_size // 4
            self.audio_buffer = self.audio_buffer[self.max_buffer_size - overlap_size:]
            
            # Process audio segment
            processed_audio = self.preprocessor.preprocess_audio(audio_segment)
            
            return processed_audio
        
        return None
    
    def reset_buffer(self):
        """Reset the audio buffer."""
        self.audio_buffer = []

