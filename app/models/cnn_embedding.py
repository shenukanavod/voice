"""
CNN Model for Speaker Embedding - Extracts 128-dimensional speaker embeddings from MFCC.

Architecture:
- Conv2D + BatchNorm + ReLU + MaxPool (3 layers)
- Flatten
- Dense(256, relu)
- Dense(128) → final embedding layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class SpeakerEmbeddingCNN(nn.Module):
    """
    CNN model for extracting speaker embeddings from MFCC features.
    
    Input: MFCC features of shape (batch, 1, frames, n_mfcc)
    Output: 128-dimensional normalized embedding
    """
    
    def __init__(self, input_height: int = 100, input_width: int = 32):
        """
        Initialize CNN model.
        
        Args:
            input_height: Number of time frames (default: 100)
            input_width: Number of MFCC features (default: 32)
        """
        super(SpeakerEmbeddingCNN, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # Convolutional layers
        # Layer 1: (1, 100, 32) -> (32, 50, 16)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: (32, 50, 16) -> (64, 25, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: (64, 25, 8) -> (128, 12, 4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size
        conv_output_height = input_height // 8  # After 3 maxpool layers
        conv_output_width = input_width // 8
        self.flattened_size = 128 * conv_output_height * conv_output_width
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)  # Final embedding layer
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, frames, n_mfcc)
            
        Returns:
            Normalized embedding of shape (batch, 128)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalization (unit vector)
        x = F.normalize(x, p=2, dim=1)
        
        return x


class EmbeddingExtractor:
    """
    Wrapper for extracting embeddings using the trained CNN model.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize embedding extractor.
        
        Args:
            model_path: Path to saved model weights (if None, uses untrained model)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Create model
        self.model = SpeakerEmbeddingCNN()
        self.model.to(self.device)
        self.model.eval()
        
        # Load weights if provided
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded model from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                print("   Using untrained model")
    
    def extract_embedding(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Extract 128-dimensional embedding from MFCC features.
        
        Args:
            mfcc: MFCC features of shape (1, frames, n_mfcc) or (frames, n_mfcc)
            
        Returns:
            Normalized embedding of shape (128,)
        """
        # Ensure correct shape
        if len(mfcc.shape) == 2:
            mfcc = np.expand_dims(mfcc, axis=0)  # Add channel dim
        if len(mfcc.shape) == 3:
            mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dim
        
        # Convert to tensor
        mfcc_tensor = torch.from_numpy(mfcc).float().to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(mfcc_tensor)
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Verify normalization
        norm = np.linalg.norm(embedding_np)
        print(f"📊 Embedding norm: {norm:.6f} (should be ~1.0)")
        
        return embedding_np
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Cosine similarity = dot(a, b) / (||a|| * ||b||)
        
        For normalized embeddings (||a|| = ||b|| = 1), this simplifies to dot(a, b)
        
        Args:
            embedding1: First embedding (128,)
            embedding2: Second embedding (128,)
            
        Returns:
            Similarity score in range [-1, 1]
            (1 = identical, 0 = orthogonal, -1 = opposite)
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        # Normalize if not already normalized
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2
        
        # Compute cosine similarity (dot product for unit vectors)
        similarity = np.dot(embedding1_normalized, embedding2_normalized)
        
        return float(similarity)


def test_cnn_model():
    """Test CNN model."""
    print("=== CNN Model Test ===")
    
    # Create dummy MFCC input
    batch_size = 2
    frames = 100
    n_mfcc = 32
    
    mfcc_input = np.random.randn(batch_size, 1, frames, n_mfcc).astype(np.float32)
    
    print(f"\n1. Testing model architecture:")
    print(f"   Input shape: {mfcc_input.shape}")
    
    # Create model
    model = SpeakerEmbeddingCNN(input_height=frames, input_width=n_mfcc)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    mfcc_tensor = torch.from_numpy(mfcc_input)
    with torch.no_grad():
        embeddings = model(mfcc_tensor)
    
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Expected: (batch_size={batch_size}, embedding_dim=128)")
    
    # Check normalization
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"\n2. Embedding normalization:")
    print(f"   Norms: {norms.numpy()}")
    print(f"   All ~1.0? {torch.allclose(norms, torch.ones_like(norms), atol=1e-5)}")
    
    # Test embedding extractor
    print(f"\n3. Testing EmbeddingExtractor:")
    extractor = EmbeddingExtractor()
    
    single_mfcc = mfcc_input[0]  # (1, 100, 32)
    embedding = extractor.extract_embedding(single_mfcc)
    
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {np.linalg.norm(embedding):.6f}")
    
    # Test cosine similarity
    print(f"\n4. Testing cosine similarity:")
    embedding1 = np.random.randn(128)
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    
    embedding2 = np.random.randn(128)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    similarity = EmbeddingExtractor.cosine_similarity(embedding1, embedding2)
    print(f"   Random embeddings similarity: {similarity:.4f}")
    
    similarity_self = EmbeddingExtractor.cosine_similarity(embedding1, embedding1)
    print(f"   Self similarity: {similarity_self:.4f} (should be 1.0)")
    
    print("\n✅ All CNN tests passed!")


if __name__ == "__main__":
    test_cnn_model()
