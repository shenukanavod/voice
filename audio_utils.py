"""
Audio Utilities for Voice Authentication
Load, preprocess, and extract features from audio files
"""

import numpy as np
import librosa
import torch
from pathlib import Path
from typing import Union, Tuple, Optional


def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    duration: Optional[float] = None,
    offset: float = 0.0,
    min_duration: float = 0.5
) -> Tuple[np.ndarray, int]:
    """
    Load audio file (.wav or .flac) and convert to mono at target sample rate.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate in Hz (default: 16000)
        duration: Duration to load in seconds (None = load full file)
        offset: Start reading after this time in seconds (default: 0.0)
        min_duration: Minimum acceptable audio duration in seconds (default: 0.5)
        
    Returns:
        Tuple of (audio_array, sample_rate)
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio format unsupported or audio too short
        RuntimeError: If audio file is corrupted
        
    Example:
        >>> audio, sr = load_audio("speech.wav", sample_rate=16000)
        >>> print(audio.shape, sr)
        (48000,) 16000
    """
    # Convert to Path object
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}\n"
            f"Please verify the file path is correct."
        )
    
    # Check file is not empty
    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Audio file is empty (0 bytes): {file_path}\n"
            f"Please provide a valid audio file."
        )
    
    # Check file extension
    if file_path.suffix.lower() not in ['.wav', '.flac']:
        raise ValueError(
            f"Unsupported audio format: {file_path.suffix}\n"
            f"Supported formats: .wav, .flac\n"
            f"Please convert your audio file to WAV or FLAC format."
        )
    
    try:
        # Load audio (librosa automatically converts to mono and resamples)
        audio, sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=True,
            duration=duration,
            offset=offset
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load audio file: {file_path}\n"
            f"Error: {str(e)}\n"
            f"The file may be corrupted or in an unsupported format."
        )
    
    # Check if audio is too short
    audio_duration = len(audio) / sr
    if audio_duration < min_duration:
        raise ValueError(
            f"Audio too short: {audio_duration:.2f}s (minimum: {min_duration:.2f}s)\n"
            f"File: {file_path}\n"
            f"Please provide audio with at least {min_duration} seconds of speech."
        )
    
    # Check for silent or very quiet audio
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        import warnings
        warnings.warn(
            f"Audio appears to be silent or very quiet (RMS: {rms:.2e})\n"
            f"File: {file_path}\n"
            f"This may result in poor verification accuracy.",
            UserWarning
        )
    
    return audio, sr


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: float = 8000.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract mel spectrogram from audio signal.
    
    Args:
        audio: Audio time series (1D numpy array)
        sample_rate: Sample rate of audio
        n_mels: Number of mel filterbanks (default: 128)
        n_fft: FFT window size (default: 2048)
        hop_length: Number of samples between successive frames (default: 512)
        fmin: Minimum frequency (Hz) (default: 20)
        fmax: Maximum frequency (Hz) (default: 8000)
        normalize: Whether to normalize the spectrogram (default: True)
        
    Returns:
        Mel spectrogram as numpy array [n_mels, time_frames]
        
    Example:
        >>> mel_spec = extract_mel_spectrogram(audio, sample_rate=16000, n_mels=128)
        >>> print(mel_spec.shape)
        (128, 94)
    """
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize
    if normalize:
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    
    return mel_db


def audio_to_tensor(
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    n_mels: int = 128,
    duration: Optional[float] = None,
    add_channel_dim: bool = True,
    add_batch_dim: bool = False
) -> torch.Tensor:
    """
    Load audio file and convert to PyTorch tensor with mel spectrogram.
    
    Complete pipeline: Load → Resample → Mono → Mel Spectrogram → Tensor
    
    Args:
        file_path: Path to audio file (.wav or .flac)
        sample_rate: Target sample rate (default: 16000 Hz)
        n_mels: Number of mel bins (default: 128)
        duration: Audio duration in seconds (None = full file)
        add_channel_dim: Add channel dimension [C, H, W] (default: True)
        add_batch_dim: Add batch dimension [B, C, H, W] (default: False)
        
    Returns:
        PyTorch tensor ready for model input
        - If add_batch_dim=True: shape [1, 1, n_mels, time]
        - If add_channel_dim=True: shape [1, n_mels, time]
        - Otherwise: shape [n_mels, time]
        
    Example:
        >>> # For single inference
        >>> tensor = audio_to_tensor("audio.wav", add_batch_dim=True)
        >>> print(tensor.shape)
        torch.Size([1, 1, 128, 94])
        
        >>> # For batch processing
        >>> tensor = audio_to_tensor("audio.wav", add_channel_dim=True)
        >>> print(tensor.shape)
        torch.Size([1, 128, 94])
    """
    # Load and preprocess audio
    audio, sr = load_audio(file_path, sample_rate=sample_rate, duration=duration)
    
    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(audio, sample_rate=sr, n_mels=n_mels)
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(mel_spec).float()
    
    # Add channel dimension if requested
    if add_channel_dim:
        tensor = tensor.unsqueeze(0)  # [1, n_mels, time]
    
    # Add batch dimension if requested
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)  # [1, 1, n_mels, time] or [1, n_mels, time]
    
    return tensor


def batch_audio_to_tensor(
    file_paths: list,
    sample_rate: int = 16000,
    n_mels: int = 128,
    duration: Optional[float] = None,
    max_length: Optional[int] = None,
    pad_mode: str = 'constant'
) -> torch.Tensor:
    """
    Load multiple audio files and convert to batched PyTorch tensor.
    
    Args:
        file_paths: List of audio file paths
        sample_rate: Target sample rate (default: 16000 Hz)
        n_mels: Number of mel bins (default: 128)
        duration: Audio duration in seconds (None = variable length)
        max_length: Maximum time frames (None = use longest in batch)
        pad_mode: Padding mode ('constant', 'reflect', 'replicate')
        
    Returns:
        Batched PyTorch tensor [batch_size, 1, n_mels, time]
        
    Example:
        >>> files = ["audio1.wav", "audio2.wav", "audio3.wav"]
        >>> batch_tensor = batch_audio_to_tensor(files)
        >>> print(batch_tensor.shape)
        torch.Size([3, 1, 128, 94])
    """
    tensors = []
    
    for file_path in file_paths:
        tensor = audio_to_tensor(
            file_path,
            sample_rate=sample_rate,
            n_mels=n_mels,
            duration=duration,
            add_channel_dim=True,
            add_batch_dim=False
        )
        tensors.append(tensor)
    
    # Find max length if not specified
    if max_length is None:
        max_length = max(t.shape[-1] for t in tensors)
    
    # Pad tensors to same length
    padded_tensors = []
    for tensor in tensors:
        if tensor.shape[-1] < max_length:
            # Pad on the right
            pad_size = max_length - tensor.shape[-1]
            if pad_mode == 'constant':
                padded = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)
            else:
                padded = torch.nn.functional.pad(tensor, (0, pad_size), mode=pad_mode)
            padded_tensors.append(padded)
        elif tensor.shape[-1] > max_length:
            # Truncate
            padded_tensors.append(tensor[..., :max_length])
        else:
            padded_tensors.append(tensor)
    
    # Stack into batch
    batch_tensor = torch.stack(padded_tensors, dim=0)
    
    return batch_tensor


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio time series
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
        
    Example:
        >>> audio_16k = resample_audio(audio_44k, orig_sr=44100, target_sr=16000)
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(
    audio: np.ndarray,
    target_level: float = -20.0
) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio time series
        target_level: Target RMS level in dB (default: -20.0)
        
    Returns:
        Normalized audio
        
    Example:
        >>> normalized_audio = normalize_audio(audio, target_level=-20.0)
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms < 1e-8:
        return audio  # Avoid division by zero for silent audio
    
    # Calculate target RMS from dB
    target_rms = 10 ** (target_level / 20.0)
    
    # Scale audio
    scaling_factor = target_rms / rms
    normalized = audio * scaling_factor
    
    return normalized


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    top_db: float = 20.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate
        top_db: Threshold in dB below reference to consider as silence
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Trimmed audio
        
    Example:
        >>> trimmed = trim_silence(audio, sample_rate=16000, top_db=20)
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return trimmed


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
        
    Example:
        >>> duration = get_audio_duration("audio.wav")
        >>> print(f"Duration: {duration:.2f} seconds")
        Duration: 3.52 seconds
    """
    duration = librosa.get_duration(path=str(file_path))
    return duration


# Convenience function for model inference
def prepare_audio_for_model(
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    n_mels: int = 128,
    duration: float = 3.0,
    device: str = 'cpu',
    check_quality: bool = True
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for model inference.
    
    Load audio → Resample to 16kHz → Convert to mono → 
    Extract mel spectrogram (128 bins) → Convert to tensor → 
    Add dimensions → Move to device
    
    Args:
        file_path: Path to audio file (.wav or .flac)
        sample_rate: Target sample rate (default: 16000 Hz)
        n_mels: Number of mel bins (default: 128)
        duration: Target duration in seconds (default: 3.0)
        device: Device to move tensor to ('cpu' or 'cuda')
        check_quality: Validate audio quality (default: True)
        
    Returns:
        PyTorch tensor ready for model [1, 1, 128, time_frames]
        
    Raises:
        FileNotFoundError: If audio file missing
        ValueError: If audio too short or invalid
        RuntimeError: If audio file corrupted
        
    Example:
        >>> tensor = prepare_audio_for_model("audio.wav", device='cuda')
        >>> output = model(tensor)
    """
    file_path = Path(file_path)
    
    # Check file exists with clear error message
    if not file_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}\n"
            f"Current directory: {Path.cwd()}\n"
            f"Please verify the file path is correct."
        )
    
    try:
        # Load and convert to tensor
        tensor = audio_to_tensor(
            file_path,
            sample_rate=sample_rate,
            n_mels=n_mels,
            duration=duration,
            add_channel_dim=True,
            add_batch_dim=True
        )
        
        # Optional quality check
        if check_quality:
            # Check for NaN or Inf values
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(
                    f"Audio contains invalid values (NaN or Inf)\n"
                    f"File: {file_path}\n"
                    f"The audio file may be corrupted."
                )
            
            # Check tensor magnitude (detect extremely quiet audio)
            tensor_max = tensor.abs().max().item()
            if tensor_max < 0.01:
                import warnings
                warnings.warn(
                    f"Audio has very low magnitude (max: {tensor_max:.4f})\n"
                    f"File: {file_path}\n"
                    f"This may indicate silent or very quiet audio.",
                    UserWarning
                )
        
        # Move to device
        tensor = tensor.to(device)
        
        return tensor
        
    except (FileNotFoundError, ValueError, RuntimeError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Wrap unexpected errors with context
        raise RuntimeError(
            f"Failed to prepare audio for model\n"
            f"File: {file_path}\n"
            f"Error: {str(e)}"
        ) from e


if __name__ == "__main__":
    # Example usage and tests
    print("Audio Utils - Example Usage")
    print("=" * 70)
    
    # Example 1: Basic loading
    print("\nExample 1: Load audio file")
    print("-" * 70)
    print("audio, sr = load_audio('audio.wav', sample_rate=16000)")
    print("# Returns: audio array and sample rate")
    
    # Example 2: Extract mel spectrogram
    print("\nExample 2: Extract mel spectrogram")
    print("-" * 70)
    print("mel_spec = extract_mel_spectrogram(audio, sample_rate=16000, n_mels=128)")
    print("# Returns: [128, time_frames] numpy array")
    
    # Example 3: Complete pipeline
    print("\nExample 3: Audio to tensor (complete pipeline)")
    print("-" * 70)
    print("tensor = audio_to_tensor('audio.wav', add_batch_dim=True)")
    print("# Returns: PyTorch tensor [1, 1, 128, time_frames]")
    
    # Example 4: Batch processing
    print("\nExample 4: Batch processing")
    print("-" * 70)
    print("files = ['audio1.wav', 'audio2.wav', 'audio3.wav']")
    print("batch = batch_audio_to_tensor(files)")
    print("# Returns: PyTorch tensor [3, 1, 128, time_frames]")
    
    # Example 5: Model inference
    print("\nExample 5: Prepare for model inference")
    print("-" * 70)
    print("tensor = prepare_audio_for_model('audio.wav', device='cuda')")
    print("# Ready for: output = model(tensor)")
    
    print("\n" + "=" * 70)
    print("✅ Audio utilities ready to use!")
