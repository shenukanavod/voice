"""
Record Additional Genuine Voice Samples
Balances the dataset by adding more real human voice samples
"""
import pyaudio
import wave
import numpy as np
from pathlib import Path
import time

# Configuration
GENUINE_DIR = Path("data/antispoofing_dataset/genuine")
SAMPLE_RATE = 16000
DURATION = 3  # seconds
CHANNELS = 1
CHUNK = 1024

# Check existing samples
existing_samples = list(GENUINE_DIR.glob("genuine_*.wav"))
start_idx = len(existing_samples)

print("="*60)
print("🎤 RECORD MORE GENUINE VOICE SAMPLES")
print("="*60)
print(f"\nCurrent genuine samples: {start_idx}")
print(f"Current spoofed samples: {len(list(Path('data/antispoofing_dataset/spoofed').glob('*.wav')))}")
print(f"\n⚠️ Dataset is imbalanced! Need more genuine samples.")
print(f"\n📝 Instructions:")
print("   - Speak naturally in different tones")
print("   - Vary your pitch and volume")
print("   - Use different phrases")
print("   - Record in different positions relative to mic")
print("="*60)

# Ask how many samples to record
while True:
    try:
        num_samples = int(input("\nHow many samples do you want to record? (recommended: 100-200): "))
        if num_samples > 0:
            break
        print("Please enter a positive number")
    except ValueError:
        print("Please enter a valid number")

print(f"\n✅ Will record {num_samples} samples")

# Sample phrases for variety
PHRASES = [
    "My voice is my password",
    "Please verify my identity",
    "Access granted to my account",
    "This is my natural voice",
    "Authentication in progress",
    "Secure voice verification",
    "Voice biometric sample",
    "Speaker recognition test",
    "I am the authorized user",
    "Grant me access please",
    "Verify my voice pattern",
    "Natural speech sample",
    "Human voice authentication",
    "This is not a recording",
    "Live voice verification"
]

# Initialize PyAudio
audio = pyaudio.PyAudio()

print("\n" + "="*60)
print("🎙️ RECORDING SESSION")
print("="*60)

for i in range(num_samples):
    print(f"\n📍 Sample {i+1}/{num_samples}")
    print(f"💬 Suggested phrase: '{PHRASES[i % len(PHRASES)]}'")
    print("   (or say anything else naturally)")
    
    input("   Press ENTER to start recording...")
    
    # Open stream
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print(f"   🔴 RECORDING for {DURATION} seconds...")
    
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    # Save WAV file
    output_file = GENUINE_DIR / f"genuine_{start_idx + i:04d}.wav"
    wf = wave.open(str(output_file), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"   ✅ Saved: {output_file.name}")

# Cleanup
audio.terminate()

print("\n" + "="*60)
print("✅ RECORDING COMPLETE!")
print("="*60)

# Final count
final_genuine = len(list(GENUINE_DIR.glob("genuine_*.wav")))
final_spoofed = len(list(Path('data/antispoofing_dataset/spoofed').glob('*.wav')))

print(f"\n📊 Updated Dataset:")
print(f"   Genuine: {final_genuine}")
print(f"   Spoofed: {final_spoofed}")
print(f"   Ratio: {final_genuine/final_spoofed*100:.1f}% genuine")

if final_genuine < 200:
    print(f"\n⚠️ Still imbalanced. Recommended: Record {200 - final_genuine} more samples")
else:
    print(f"\n✅ Good balance! Ready to retrain.")

print(f"\n🚀 Next step: Retrain the model")
print("   Run: python train_antispoofing.py --epochs 25 --batch-size 32")
print("="*60)
