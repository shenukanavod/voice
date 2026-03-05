"""
Voice Authentication System
Clean, modular implementation for speaker verification
"""

from .voice_auth import VoiceAuthSystem
from .model_utils import EmbeddingExtractor, CNNLSTMEmbedding
from .audio_processor import AudioProcessor
from .profile_manager import VoiceProfileManager

__all__ = [
    'VoiceAuthSystem',
    'EmbeddingExtractor',
    'CNNLSTMEmbedding',
    'AudioProcessor',
    'VoiceProfileManager'
]

__version__ = '1.0.0'
