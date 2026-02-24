"""
Model Module
CNN + LSTM architecture for speaker embedding extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


class CNNLSTMEmbedding(nn.Module):
    """CNN-LSTM model for speaker embedding extraction."""

    def __init__(self, embedding_dim: int = 128, n_mels: int = 120):
        super(CNNLSTMEmbedding, self).__init__()

        self.embedding_dim = embedding_dim

        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )

        # LSTM
        self.lstm_input_size = 256 * (n_mels // 8)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        """Forward pass: CNN → LSTM → Attention → Embedding."""
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

        # Embedding
        embeddings = self.embedding(pooled)

        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


def load_model(
    model_path: str, device: str = "cpu"
) -> tuple[CNNLSTMEmbedding, bool]:
    """
    Load CNN+LSTM model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        (model, is_trained): Model and whether it was successfully trained
    """
    device = torch.device(device)
    
    if not Path(model_path).exists():
        print(f"⚠️  Model path not found: {model_path}")
        print("   Using untrained model with 128D embeddings")
        model = CNNLSTMEmbedding(embedding_dim=128, n_mels=120)
        model.to(device)
        model.eval()
        return model, False

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Auto-detect embedding_dim from checkpoint config
        embedding_dim = 128
        n_mels = 120
        if "config" in checkpoint:
            embedding_dim = checkpoint["config"].get("embedding_dim", 128)
            n_mels = checkpoint["config"].get("n_mels", 120)
        
        # Create model with correct dimensions
        model = CNNLSTMEmbedding(embedding_dim=embedding_dim, n_mels=n_mels)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print(f"✅ Loaded CNN+LSTM model from {model_path}")
        print(f"   Embedding dim: {embedding_dim}, n_mels: {n_mels}")
        return model, True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Using untrained model with 128D embeddings")
        model = CNNLSTMEmbedding(embedding_dim=128, n_mels=120)
        model.to(device)
        model.eval()
        return model, False
