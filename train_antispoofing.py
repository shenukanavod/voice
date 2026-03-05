#!/usr/bin/env python3
"""
Anti-Spoofing Model Training Script

Trains the CNN-based anti-spoofing model using generated or downloaded dataset.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.models.anti_spoofing import AntiSpoofingCNN


class AntiSpoofingDataset(Dataset):
    """PyTorch Dataset for anti-spoofing training"""
    
    def __init__(self, audio_files, labels, transform=None):
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=16000,
                n_mels=128,
                fmax=8000,
                n_fft=512,
                hop_length=256
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Ensure fixed size (pad or truncate)
            target_length = 128
            if mel_spec_db.shape[1] < target_length:
                # Pad
                pad_width = target_length - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncate or take random crop during training
                if self.transform == 'train':
                    start = np.random.randint(0, mel_spec_db.shape[1] - target_length + 1)
                    mel_spec_db = mel_spec_db[:, start:start+target_length]
                else:
                    # Center crop for validation
                    start = (mel_spec_db.shape[1] - target_length) // 2
                    mel_spec_db = mel_spec_db[:, start:start+target_length]
            
            # Convert to tensor
            mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)  # Add channel dimension
            
            return mel_spec_tensor, label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros on error
            return torch.zeros(1, 128, 128), label


def load_dataset(dataset_dir='data/antispoofing_dataset'):
    """
    Load dataset from directory
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        Lists of file paths and labels
    """
    dataset_path = Path(dataset_dir)
    genuine_dir = dataset_path / 'genuine'
    spoofed_dir = dataset_path / 'spoofed'
    
    audio_files = []
    labels = []
    
    # Load genuine samples (label = 0)
    if genuine_dir.exists():
        for file in genuine_dir.glob('*.wav'):
            audio_files.append(str(file))
            labels.append(0)  # 0 = genuine
    
    # Load spoofed samples (label = 1)
    if spoofed_dir.exists():
        for file in spoofed_dir.glob('*.wav'):
            audio_files.append(str(file))
            labels.append(1)  # 1 = spoofed
    
    return audio_files, labels


def train_model(
    dataset_dir='data/antispoofing_dataset',
    epochs=30,
    batch_size=32,
    learning_rate=0.001,
    model_save_path='models/anti_spoofing_model.pth'
):
    """
    Train anti-spoofing model
    
    Args:
        dataset_dir: Path to dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_save_path: Where to save trained model
    """
    print("=" * 60)
    print("🎯 ANTI-SPOOFING MODEL TRAINING")
    print("=" * 60)
    
    # Load dataset
    print("\n📁 Loading dataset...")
    audio_files, labels = load_dataset(dataset_dir)
    
    if len(audio_files) == 0:
        print("❌ No data found! Please generate dataset first.")
        print("   Run: python generate_spoofing_dataset.py")
        return
    
    print(f"✅ Loaded {len(audio_files)} samples")
    print(f"   Genuine: {labels.count(0)}")
    print(f"   Spoofed: {labels.count(1)}")
    
    # Split dataset
    print("\n📊 Splitting dataset...")
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"   Training samples: {len(train_files)}")
    print(f"   Validation samples: {len(val_files)}")
    
    # Create datasets and dataloaders
    train_dataset = AntiSpoofingDataset(train_files, train_labels, transform='train')
    val_dataset = AntiSpoofingDataset(val_files, val_labels, transform='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n🤖 Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = AntiSpoofingCNN(input_channels=1, num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for spectrograms, labels_batch in train_pbar:
            spectrograms = spectrograms.to(device)
            labels_batch = torch.LongTensor(labels_batch).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            for spectrograms, labels_batch in val_pbar:
                spectrograms = spectrograms.to(device)
                labels_batch = torch.LongTensor(labels_batch).to(device)
                
                outputs = model(spectrograms)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Calculate detailed metrics
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print("-" * 60)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"💾 New best model! Saving to {model_save_path}")
            
            # Create models directory if needed
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, model_save_path)
    
    # Training complete
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_save_path}")
    
    # Save training history
    history_path = Path(model_save_path).parent / 'antispoofing_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Plot training curves
    plot_training_history(history, Path(model_save_path).parent / 'training_curves.png')
    
    return model, history


def plot_training_history(history, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training curves saved to: {save_path}")
    plt.close()


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Anti-Spoofing Model')
    parser.add_argument('--dataset', type=str, default='data/antispoofing_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='models/anti_spoofing_model.pth',
                        help='Output model path')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_save_path=args.output
    )


if __name__ == "__main__":
    main()
