"""
Noise Filter Module - Removes background noise using spectral subtraction and Wiener filtering.
"""

import numpy as np
import librosa
from scipy import signal
from typing import Optional


class NoiseFilter:
    """
    Advanced noise removal using spectral subtraction and Wiener filtering.
    """
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 256):
        """
        Initialize noise filter.
       
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def remove_noise(self, audio: np.ndarray, noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove noise from audio using spectral subtraction + Wiener filter.
  
        """
        print("Applying noise removal...")
        
        # Step 1: Spectral subtraction
        audio_cleaned = self._spectral_subtraction(audio, noise_profile)
        
        # Step 2: Wiener filtering
        audio_final = self._wiener_filter(audio_cleaned)
        
        # Step 3: Ensure no clipping
        audio_final = np.clip(audio_final, -1.0, 1.0)
        
        print(" Noise removal complete")
        return audio_final
    
    def _spectral_subtraction(self, audio: np.ndarray, noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
         removes stationary background noise.
     
        
        
        """
        # Compute Short-Time Fourier Transform
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum (from first 0.5 seconds if not provided)
        if noise_profile is None:
            noise_frames = int(0.5 * self.sample_rate / self.hop_length)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        else:
            noise_spectrum = noise_profile
        
        # Spectral subtraction with over-subtraction factor
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Floor factor to prevent musical noise
        
        # Subtract noise spectrum
        magnitude_cleaned = magnitude - alpha * noise_spectrum
        
        # Apply spectral floor (prevents negative values and musical noise)
        magnitude_cleaned = np.maximum(magnitude_cleaned, beta * magnitude)
        
        # Reconstruct audio
        stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
        audio_cleaned = librosa.istft(stft_cleaned, hop_length=self.hop_length, length=len(audio))
        
        return audio_cleaned
    
    def _wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Wiener filter - adaptive noise reduction.
        
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise power (from weakest 10% of frames)
        noise_power = np.percentile(magnitude ** 2, 10, axis=1, keepdims=True)
        
        # Estimate signal power
        signal_power = magnitude ** 2
        
        # Compute Wiener filter gain
        # G = signal_power / (signal_power + noise_power)
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
        
        # Apply filter
        magnitude_filtered = magnitude * wiener_gain
        
        # Reconstruct audio
        stft_filtered = magnitude_filtered * np.exp(1j * phase)
        audio_filtered = librosa.istft(stft_filtered, hop_length=self.hop_length, length=len(audio))
        
        return audio_filtered
    
    def estimate_noise_profile(self, noise_audio: np.ndarray) -> np.ndarray:
        """
        Estimate noise profile from a noise-only audio sample.
        
        """
        stft = librosa.stft(noise_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        noise_profile = np.mean(magnitude, axis=1, keepdims=True)
        
        return noise_profile
    
    def apply_bandpass_filter(self, audio: np.ndarray, 
                              lowcut: float = 80.0, 
                              highcut: float = 3400.0) -> np.ndarray:
        """
        Apply bandpass filter to keep only human voice frequencies.
        
        """
        # Design Butterworth bandpass filter
        nyquist = self.sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # 4th order Butterworth filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        audio_filtered = signal.filtfilt(b, a, audio)
        
        return audio_filtered
    
    def remove_clicks_and_pops(self, audio: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Remove clicks and pops (impulse noise).
        

        """
        # Calculate local statistics
        window_size = int(0.01 * self.sample_rate)  # 10ms window
        audio_cleaned = audio.copy()
        
        for i in range(window_size, len(audio) - window_size):
            # Get local window
            window = audio[i - window_size:i + window_size]
            local_mean = np.mean(window)
            local_std = np.std(window)
            
            # Detect outliers (clicks/pops)
            if np.abs(audio[i] - local_mean) > threshold * local_std:
                # Replace with linear interpolation
                audio_cleaned[i] = (audio[i - 1] + audio[i + 1]) / 2
        
        return audio_cleaned


def test_noise_filter():
    """Test noise filtering on sample audio."""
    print("=== Noise Filter Test ===")
    
    # Generate test signal: sine wave + noise
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean signal (440 Hz sine wave)
    clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Add noise
    noise = 0.1 * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # Apply noise filter
    noise_filter = NoiseFilter(sample_rate=sample_rate)
    filtered_signal = noise_filter.remove_noise(noisy_signal)
    
    # Calculate SNR improvement
    noise_power_before = np.mean((noisy_signal - clean_signal) ** 2)
    noise_power_after = np.mean((filtered_signal - clean_signal) ** 2)
    
    snr_improvement = 10 * np.log10(noise_power_before / (noise_power_after + 1e-10))
    
    print(f" SNR improvement: {snr_improvement:.2f} dB")
    
    # Apply bandpass filter
    bandpass_signal = noise_filter.apply_bandpass_filter(filtered_signal)
    print(f" Bandpass filter applied (80-3400 Hz)")


if __name__ == "__main__":
    test_noise_filter()
