#!/usr/bin/env python3
"""
Record Genuine Voice Samples for Anti-Spoofing Training

Records live microphone input to create genuine voice training data.
"""

import pyaudio
import wave
import numpy as np
from pathlib import Path
import time

def record_training_samples(num_users=5, samples_per_user=3):
    """
    Record voice samples for anti-spoofing training
    
    Args:
        num_users: Number of different speakers
        samples_per_user: Samples per speaker
    """
    print("=" * 60)
    print("🎤 RECORD GENUINE VOICE SAMPLES FOR TRAINING")
    print("=" * 60)
    print(f"\nWe'll record {num_users} speakers, {samples_per_user} samples each")
    print(f"Total: {num_users * samples_per_user} genuine voice samples\n")
    
    # Setup
    genuine_dir = Path("data/antispoofing_dataset/genuine")
    genuine_dir.mkdir(parents=True, exist_ok=True)
    
    # PyAudio setup
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    
    p = pyaudio.PyAudio()
    
    sample_count = 0
    
    try:
        for user_num in range(1, num_users + 1):
            print(f"\n{'='*60}")
            print(f"📢 SPEAKER {user_num} of {num_users}")
            print(f"{'='*60}")
            input(f"Press ENTER when Speaker {user_num} is ready...")
            
            for sample_num in range(1, samples_per_user + 1):
                print(f"\n🎙️  Recording Sample {sample_num}/{samples_per_user}...")
                print("   Say something like:")
                print("   - 'My voice is my password'")
                print(f"   - 'This is speaker {user_num} sample {sample_num}'")
                print("   - Or any natural speech\n")
                
                input("   Press ENTER to start recording...")
                
                # Record
                print(f"   🔴 RECORDING for {RECORD_SECONDS} seconds...")
                stream = p.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)
                
                frames = []
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                
                print("   ✅ Recording complete!")
                
                # Save
                output_file = genuine_dir / f"genuine_{sample_count:04d}.wav"
                wf = wave.open(str(output_file), 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                print(f"   💾 Saved: {output_file.name}")
                sample_count += 1
                
                time.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("✅ RECORDING COMPLETE!")
        print("=" * 60)
        print(f"📊 Total genuine samples recorded: {sample_count}")
        print(f"📁 Location: {genuine_dir.absolute()}")
        print("\n🚀 Next step: Retrain the model")
        print("   Run: python train_antispoofing.py --epochs 20")
        print("=" * 60)
        
    finally:
        p.terminate()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Record voice samples for anti-spoofing training')
    parser.add_argument('--users', type=int, default=5, help='Number of speakers (default: 5)')
    parser.add_argument('--samples', type=int, default=3, help='Samples per speaker (default: 3)')
    
    args = parser.parse_args()
    
    record_training_samples(num_users=args.users, samples_per_user=args.samples)
