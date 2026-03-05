"""
24-HOUR OPTIMIZED TRAINING SCRIPT
Designed for CPU training with time constraints
Target: 70-80% accuracy in 24 hours
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import librosa
import random
from tqdm import tqdm
from datetime import datetime, timedelta
import json
import time

from model import CNNLSTMEmbedding
from preprocessing import AudioPreprocessor


class OptimizedVoiceDataset(Dataset):
    """Optimized dataset loader for fast CPU training - Supports VoxCeleb structure"""
    
    def __init__(self, root_dir, max_speakers=400, samples_per_speaker=120, 
                 split='train', train_ratio=0.8, seed=42):
        """
        Args:
            root_dir: Root directory with speaker folders (VoxCeleb format)
            max_speakers: Maximum speakers to load (for 24hr training)
            samples_per_speaker: Max samples per speaker
            split: 'train' or 'val'
            train_ratio: Train/val split ratio
            seed: Random seed
        """
        self.root_dir = Path(root_dir)
        self.preprocessor = AudioPreprocessor(sample_rate=16000)
        
        print(f"\n📂 Loading VoxCeleb dataset from: {root_dir}")
        print(f"   Target: {max_speakers} speakers, up to {samples_per_speaker} samples each")
        
        # Find all speaker folders
        speaker_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        print(f"   Found {len(speaker_folders)} total speakers")
        
        # Limit speakers for 24-hour training
        random.seed(seed)
        if len(speaker_folders) > max_speakers:
            speaker_folders = random.sample(speaker_folders, max_speakers)
            print(f"   ✂️  Limited to {max_speakers} speakers for 24-hour training")
        
        # Load audio file paths from VoxCeleb structure (speaker/video/audio.wav)
        self.data = []
        for speaker_id, speaker_dir in enumerate(tqdm(speaker_folders, desc="Loading speakers")):
            audio_files = []
            
            # VoxCeleb has nested structure: speaker/video/audio.wav
            for video_dir in speaker_dir.iterdir():
                if video_dir.is_dir():
                    audio_files.extend(list(video_dir.glob("*.wav")))
                    audio_files.extend(list(video_dir.glob("*.flac")))
            
            # Limit samples per speaker
            if len(audio_files) > samples_per_speaker:
                audio_files = random.sample(audio_files, samples_per_speaker)
            
            for audio_path in audio_files:
                self.data.append((str(audio_path), speaker_id))
        
        # Split train/val
        random.seed(seed)
        random.shuffle(self.data)
        split_idx = int(len(self.data) * train_ratio)
        
        if split == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
        
        self.speakers = len(speaker_folders)
        print(f"✅ Loaded {len(self.data)} samples from {self.speakers} speakers")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get triplet: anchor, positive, negative"""
        # Anchor
        anchor_path, anchor_speaker = self.data[idx]
        anchor_mfcc = self._load_audio(anchor_path)
        
        # Positive (same speaker)
        positive_samples = [path for path, spk in self.data if spk == anchor_speaker and path != anchor_path]
        if positive_samples:
            positive_path = random.choice(positive_samples)
            positive_mfcc = self._load_audio(positive_path)
        else:
            positive_mfcc = anchor_mfcc
        
        # Negative (different speaker)
        negative_samples = [path for path, spk in self.data if spk != anchor_speaker]
        negative_path = random.choice(negative_samples)
        negative_mfcc = self._load_audio(negative_path)
        
        return anchor_mfcc, positive_mfcc, negative_mfcc, anchor_speaker
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000, duration=5.0)
            audio = self.preprocessor.preprocess(audio)
            mfcc = self.preprocessor.extract_mfcc(audio)
            mfcc = self.preprocessor.pad_mfcc(mfcc, target_frames=100)
            return torch.from_numpy(mfcc).float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(120, 100)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for anchor, positive, negative, _ in pbar:
        anchor = anchor.unsqueeze(1).to(device)
        positive = positive.unsqueeze(1).to(device)
        negative = negative.unsqueeze(1).to(device)
        
        # Forward
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            pos_dist = torch.norm(anchor_emb - positive_emb, dim=1)
            neg_dist = torch.norm(anchor_emb - negative_emb, dim=1)
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for anchor, positive, negative, _ in tqdm(val_loader, desc="Validating"):
            anchor = anchor.unsqueeze(1).to(device)
            positive = positive.unsqueeze(1).to(device)
            negative = negative.unsqueeze(1).to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
            
            pos_dist = torch.norm(anchor_emb - positive_emb, dim=1)
            neg_dist = torch.norm(anchor_emb - negative_emb, dim=1)
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)
    
    return total_loss / len(val_loader), 100 * correct / total


def estimate_time_remaining(start_time, current_epoch, total_epochs):
    """Estimate remaining training time"""
    elapsed = time.time() - start_time
    avg_time_per_epoch = elapsed / current_epoch if current_epoch > 0 else 0
    remaining_epochs = total_epochs - current_epoch
    estimated_remaining = avg_time_per_epoch * remaining_epochs
    
    return timedelta(seconds=int(estimated_remaining))


def main():
    """24-Hour Optimized Training"""
    
    print("\n" + "="*80)
    print(" "*20 + "24-HOUR OPTIMIZED TRAINING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Configuration for 30-hour CPU training (OPTIMIZED FOR YOUR DATASET)
    config = {
        'data_root': 'dataset',  # VoxCeleb dataset folder
        'max_speakers': 600,  # 600 speakers = ~30 hours training time
        'samples_per_speaker': 120,  # Use all available samples per speaker
        'batch_size': 16,  # CPU-friendly batch size
        'embedding_dim': 256,
        'n_mels': 120,
        'learning_rate': 0.0003,
        'num_epochs': 18,  # 18 epochs achievable in ~30 hours on CPU
        'margin': 0.5,
        'train_ratio': 0.8,
        'model_save_path': 'models/trained_30hour_600spk.pth',
        'seed': 42
    }
    
    # Dataset info: 1,211 speakers, 148,642 files, ~105 files/speaker
    # Training with 600 speakers = ~63,240 files
    # Expected: 80-84% accuracy in 30 hours
    
    # Check dataset path
    if not Path(config['data_root']).exists():
        print(f"\n❌ ERROR: Dataset path not found: {config['data_root']}")
        print(f"   Expected path: C:\\Users\\user\\Desktop\\voice - Copy - Copy\\dataset")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Training Device: {device}")
    if device.type == 'cpu':
        print("   ⚠️  CPU training detected - estimated time: 30 hours")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = OptimizedVoiceDataset(
        root_dir=config['data_root'],
        max_speakers=config['max_speakers'],
        samples_per_speaker=config['samples_per_speaker'],
        split='train',
        train_ratio=config['train_ratio'],
        seed=config['seed']
    )
    
    val_dataset = OptimizedVoiceDataset(
        root_dir=config['data_root'],
        max_speakers=config['max_speakers'],
        samples_per_speaker=config['samples_per_speaker'],
        split='val',
        train_ratio=config['train_ratio'],
        seed=config['seed']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = CNNLSTMEmbedding(embedding_dim=config['embedding_dim'], 
                            n_mels=config['n_mels']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.TripletMarginLoss(margin=config['margin'], p=2)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    print("\n🚀 Starting training...")
    print(f"⏱️  Target: {config['num_epochs']} epochs in ~30 hours\n")
    
    start_time = time.time()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Early stopping and plateau detection
    patience = 5  # Number of epochs to wait for improvement
    plateau_threshold = 0.5  # Min accuracy change to consider improvement
    epochs_without_improvement = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Time estimates
        epoch_time = time.time() - epoch_start
        remaining_time = estimate_time_remaining(start_time, epoch, config['num_epochs'])
        total_elapsed = timedelta(seconds=int(time.time() - start_time))
        
        # Log
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time/60:.1f} min | Elapsed: {total_elapsed}")
        print(f"Remaining:  ~{remaining_time} (estimated)")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0  # Reset counter
            Path(config['model_save_path']).parent.mkdir(exist_ok=True, parents=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'train_accuracy': train_acc,
                'config': config,
                'history': history
            }, config['model_save_path'])
            
            print(f"💾 Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            
            # Check if accuracy is stuck
            if epochs_without_improvement >= 3:
                print(f"⚠️  No improvement for {epochs_without_improvement} epochs")
                
                if epochs_without_improvement >= 5:
                    print(f"⚠️  WARNING: Accuracy stuck at {val_acc:.2f}% for 5 epochs")
                    print(f"   Possible solutions:")
                    print(f"   1. Learning rate might be too low (current: {optimizer.param_groups[0]['lr']:.6f})")
                    print(f"   2. Model might need more data (increase max_speakers)")
                    print(f"   3. Continue training - might improve later")
        
        # Check if accuracy is too low after several epochs
        if epoch >= 5 and val_acc < 55:
            print(f"\n⚠️  WARNING: Validation accuracy is low ({val_acc:.2f}%) after {epoch} epochs")
            print(f"   This might indicate:")
            print(f"   1. Learning rate too high or too low")
            print(f"   2. Insufficient data diversity")
            print(f"   3. Model architecture issues")
            print(f"   Recommendation: Continue training to epoch 10 and reassess")
        
        # Encouraging messages for normal progress
        if epoch >= 3 and val_acc > 60:
            if epoch == 3:
                print(f"✅ Good progress! Model is learning effectively")
        
        if epoch >= 6 and val_acc > 70:
            if epoch == 6:
                print(f"✅ Excellent! On track for 80%+ accuracy")
        
        if epoch >= 10 and val_acc > 75:
            if epoch == 10:
                print(f"🎯 Great progress! Target accuracy likely achievable")
        
        print(f"{'='*80}\n")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(" "*25 + "TRAINING COMPLETE")
    print("="*80)
    print(f"Total Time: {timedelta(seconds=int(total_time))}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Model saved: {config['model_save_path']}")
    print("="*80)
    
    # Save training history
    history_path = Path(config['model_save_path']).parent / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {history_path}")


if __name__ == "__main__":
    main()
