"""
Model Utilities for Voice Authentication
Handles model loading and embedding extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import numpy as np


class CNNLSTMEmbedding(nn.Module):
    """CNN-LSTM model for speaker embedding extraction."""
    
    def __init__(self, embedding_dim: int = 256, n_mels: int = 120):
        super(CNNLSTMEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # LSTM
        self.lstm_input_size = 256 * (n_mels // 8)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted = lstm_out * attention_weights
        pooled = weighted.sum(dim=1)
        
        # Embeddings
        embeddings = self.embedding(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class EmbeddingExtractor:
    """Wrapper for loading model and extracting embeddings."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
       
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Embedding dimension: {self.model.embedding_dim}")
    
    def _load_model(self) -> CNNLSTMEmbedding:
        """Load model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        embedding_dim = config.get('embedding_dim', 256)
        n_mels = config.get('n_mels', 120)
        
        # Initialize model
        model = CNNLSTMEmbedding(embedding_dim=embedding_dim, n_mels=n_mels)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def extract_embedding(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Extract embedding from mel spectrogram.
        
    
        """
        with torch.no_grad():
            # Convert to tensor
            if isinstance(mel_spectrogram, np.ndarray):
                mel_tensor = torch.from_numpy(mel_spectrogram).float()
            else:
                mel_tensor = mel_spectrogram
            
            # Add batch and channel dimensions
            if mel_tensor.dim() == 2:
                mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)
            elif mel_tensor.dim() == 3:
                mel_tensor = mel_tensor.unsqueeze(0)
            
            # Move to device
            mel_tensor = mel_tensor.to(self.device)
            
            # Extract embedding
            embedding = self.model(mel_tensor)
            
            # Convert to numpy
            embedding = embedding.cpu().numpy().squeeze()
            
            return embedding
    
    def extract_embeddings_batch(self, mel_spectrograms: list) -> np.ndarray:
        """
        Extract embeddings from multiple mel spectrograms.
        
        """
        embeddings = []
        
        for mel in mel_spectrograms:
            emb = self.extract_embedding(mel)
            embeddings.append(emb)
        
        return np.array(embeddings)
