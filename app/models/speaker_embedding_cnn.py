"""
Speaker Embedding CNN Model 
Audio → MFCC → CNN → 128D Embeddings → Authentication / This model learns discriminative speaker embeddings through triplet loss training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEmbeddingCNN(nn.Module):
    """
    CNN that takes MFCC features and outputs speaker embeddings.
    
    Flow: MFCC (n_mels × time) → CNN layers → 128D embedding
    
    Uses triplet loss during training to learn discriminative embeddings:
    - Anchor and Positive (same speaker) should be close
    - Anchor and Negative (different speakers) should be far apart
    """
    
    def __init__(self, n_mels=64, embedding_dim=128, dropout=0.3):
        super(SpeakerEmbeddingCNN, self).__init__()
        
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # Block 1: 1 → 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Reduces spatial dimensions by half
            nn.Dropout2d(dropout * 0.5),
            
            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),
            
            # Block 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Block 4: 128 → 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling → 256 features
        )
        
        # Embedding layers
        self.embedding_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input MFCC tensor (batch_size, 1, n_mels, time_frames)
        
        Returns:
            embeddings: L2-normalized embeddings (batch_size, embedding_dim)
        """
        # Extract features
        features = self.conv_layers(x)
        
        # Generate embeddings
        embeddings = self.embedding_layers(features)
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def extract_embedding(self, mfcc_features):
        """
        Extract speaker embedding from MFCC features.
        
        Args:
            mfcc_features: numpy array (n_mels, time_frames)
        
        Returns:
            embedding: numpy array (embedding_dim,)
        """
        # Prepare input
        x = torch.from_numpy(mfcc_features).float()
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Extract embedding
        self.eval()
        with torch.no_grad():
            embedding = self.forward(x)
        
        return embedding.squeeze(0).cpu().numpy()


class TripletLoss(nn.Module):
    """
    Triplet Loss for training speaker embeddings.
    
    Loss = max(0, margin + d(anchor, positive) - d(anchor, negative))
    
    Where d() is distance function (1 - cosine_similarity)
    """
    
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Calculate triplet loss
        
        Args:
            anchor: Embeddings from anchor samples (batch_size, embedding_dim)
            positive: Embeddings from positive samples (same speaker as anchor)
            negative: Embeddings from negative samples (different speaker)
        
        Returns:
            loss: Scalar triplet loss
        """
        # Calculate distances (using cosine distance = 1 - cosine_similarity)
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        
        # Triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        return losses.mean()


def create_model(n_mels=64, embedding_dim=128, dropout=0.3):
    """
    Factory function to create speaker embedding CNN.
    
    Args:
        n_mels: Number of mel frequency bins
        embedding_dim: Dimension of output embeddings
        dropout: Dropout rate for regularization
    
    Returns:
        model: SpeakerEmbeddingCNN instance
    """
    model = SpeakerEmbeddingCNN(n_mels, embedding_dim, dropout)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    return model


if __name__ == "__main__":
    # Test model
    model = create_model(n_mels=64, embedding_dim=128)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    n_mels = 64
    time_frames = 128
    
    x = torch.randn(batch_size, 1, n_mels, time_frames)
    embeddings = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding L2 norms: {torch.norm(embeddings, dim=1)}")  # Should be ~1.0
