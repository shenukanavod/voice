#!/usr/bin/env python3
"""
Retrain Anti-Spoofing with User's Actual Microphone Data

Uses enrolled user voices as genuine samples for retraining.
"""

import shutil
from pathlib import Path
import os

def retrain_with_user_voices():
    """Copy enrolled user voices to genuine dataset and retrain"""
    
    print("=" * 60)
    print("🔄 RETRAINING ANTI-SPOOFING WITH YOUR MICROPHONE DATA")
    print("=" * 60)
    
    # Directories
    voice_profiles_dir = Path("data/voice_profiles")
    genuine_dir = Path("data/antispoofing_dataset/genuine")
    
    # Clear old LibriSpeech samples
    print("\n🧹 Clearing old training data...")
    if genuine_dir.exists():
        shutil.rmtree(genuine_dir)
    genuine_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy enrolled user samples
    print("\n📁 Collecting enrolled user voice samples...")
    
    sample_count = 0
    
    # Look for WAV files in voice profiles
    for wav_file in voice_profiles_dir.rglob("*.wav"):
        try:
            # Copy to genuine dataset
            dest = genuine_dir / f"genuine_{sample_count:04d}.wav"
            shutil.copy2(wav_file, dest)
            sample_count += 1
            print(f"   Copied: {wav_file.name}")
        except Exception as e:
            print(f"   Error copying {wav_file.name}: {e}")
    
    print(f"\n✅ Collected {sample_count} genuine samples from enrolled users")
    
    if sample_count < 10:
        print("\n⚠️  WARNING: Very few samples!")
        print("   Recommendation: Enroll at least 3-5 users first")
        print("   Each user provides multiple samples during enrollment")
        return False
    
    print("\n🚀 Ready to retrain!")
    print(f"   Genuine samples (your mic): {sample_count}")
    print(f"   Spoofed samples (TTS): 200")
    print("\n📝 Next step:")
    print("   Run: python train_antispoofing.py --epochs 20")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = retrain_with_user_voices()
    
    if success:
        print("\n✅ Data prepared! Now run:")
        print("   python train_antispoofing.py --epochs 20")
    else:
        print("\n❌ Not enough data. Please:")
        print("   1. Run the UI: python desktop_app.py")
        print("   2. Enroll 3-5 users")
        print("   3. Run this script again")
