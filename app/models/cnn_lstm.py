import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN model for voice authentication with embedding extraction.
class VoiceAuthCNN(nn.Module): 
    
    
    def __init__(self, num_classes=50, n_mels=64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv block 1 Learns basic voice patterns
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Conv block 2 Learns more complex speaker features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Conv block 3 Learns high-level voice identity features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Conv block 4 Compresses entire spectrogram
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Match original architecture from training script
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass for classification."""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def get_embedding(self, x):
        """Extract embedding (feature vector before final classification layer)."""
        # Pass through convolutional layers
        x = self.conv_layers(x)
        
        # Pass through FC layers up to the penultimate layer (128-dim)
        # This gives us the discriminative embedding before classification
        x = nn.Flatten()(x)
        x = self.fc_layers[1](x)  # First Linear: 256 -> 256
        x = self.fc_layers[2](x)  # ReLU
        x = self.fc_layers[3](x)  # Dropout
        x = self.fc_layers[4](x)  # Second Linear: 256 -> 128
        # REMOVED ReLU HERE - extract before ReLU to get dense embeddings (not sparse)
        # Don't apply the final dropout and classification layer
        
        # L2 normalize the embedding for better discrimination
        # This makes cosine similarity more meaningful
        x = nn.functional.normalize(x, p=2, dim=1)
        
        return x
