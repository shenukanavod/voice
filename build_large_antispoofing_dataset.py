"""
Build Large Anti-Spoofing Dataset
Downloads genuine voices and generates diverse spoofed samples
"""
import os
import urllib.request
import tarfile
import shutil
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from gtts import gTTS
from tqdm import tqdm
import random

# Dataset configuration
DATASET_DIR = Path("data/antispoofing_dataset")
GENUINE_DIR = DATASET_DIR / "genuine"
SPOOFED_DIR = DATASET_DIR / "spoofed"
TEMP_DIR = Path("data/temp_download")

# Target counts
TARGET_GENUINE = 500  # Download 500 genuine samples
TARGET_SPOOFED = 500  # Generate 500 spoofed samples

# Create directories
GENUINE_DIR.mkdir(parents=True, exist_ok=True)
SPOOFED_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("🎯 BUILDING LARGE ANTI-SPOOFING DATASET")
print("="*60)

# ============================================================
# PART 1: Download Genuine Voice Samples
# ============================================================
print("\n📥 PART 1: Downloading Genuine Voice Samples...")
print(f"Target: {TARGET_GENUINE} samples")

def download_librispeech_samples():
    """Download LibriSpeech dev-clean subset (small, ~350MB)"""
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = TEMP_DIR / "dev-clean.tar.gz"
    
    print(f"\n📦 Downloading LibriSpeech dev-clean...")
    print(f"URL: {url}")
    print(f"Size: ~350MB")
    
    # Download with progress
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r  Progress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)", end="")
        
        urllib.request.urlretrieve(url, tar_path, reporthook=progress_hook)
        print("\n✅ Download complete!")
        
        # Extract
        print("📂 Extracting files...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(TEMP_DIR)
        print("✅ Extraction complete!")
        
        return TEMP_DIR / "LibriSpeech" / "dev-clean"
    
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        return None

def process_librispeech_files(source_dir, target_count):
    """Convert LibriSpeech FLAC files to WAV and move to genuine folder"""
    if not source_dir or not source_dir.exists():
        print("⚠️ Source directory not found")
        return 0
    
    print(f"\n🔄 Processing FLAC files...")
    
    # Find all FLAC files
    flac_files = list(source_dir.rglob("*.flac"))
    print(f"Found {len(flac_files)} FLAC files")
    
    # Check existing genuine samples
    existing_samples = list(GENUINE_DIR.glob("genuine_*.wav"))
    start_idx = len(existing_samples)
    print(f"Existing genuine samples: {start_idx}")
    
    # Process limited number
    count = 0
    for flac_file in tqdm(flac_files[:target_count], desc="Converting"):
        try:
            # Load FLAC
            audio, sr = librosa.load(flac_file, sr=16000, mono=True)
            
            # Save as WAV
            output_file = GENUINE_DIR / f"genuine_{start_idx + count:04d}.wav"
            sf.write(output_file, audio, 16000)
            count += 1
            
        except Exception as e:
            print(f"\n⚠️ Error processing {flac_file.name}: {e}")
            continue
    
    print(f"\n✅ Processed {count} genuine samples")
    return count

# Download and process
librispeech_dir = download_librispeech_samples()
genuine_count = process_librispeech_files(librispeech_dir, TARGET_GENUINE)

# ============================================================
# PART 2: Generate Diverse Spoofed Samples
# ============================================================
print("\n🎭 PART 2: Generating Spoofed Voice Samples...")
print(f"Target: {TARGET_SPOOFED} samples")

# Check existing spoofed samples
existing_spoofed = list(SPOOFED_DIR.glob("spoofed_*.wav"))
start_idx = len(existing_spoofed)
print(f"Existing spoofed samples: {start_idx}")

# Diverse text prompts for TTS
PHRASES = [
    "My voice is my password verify me",
    "This is a test of the voice authentication system",
    "Please authenticate my voice",
    "I would like to access my account",
    "Voice recognition test sample",
    "Security verification in progress",
    "Open sesame voice authentication",
    "Artificial intelligence voice sample",
    "Neural network generated speech",
    "Machine learning audio test",
    "Deep learning voice synthesis",
    "Computer generated speech pattern",
    "Automated voice authentication",
    "Synthetic speech generation test",
    "Text to speech conversion sample"
]

# Different TTS languages for variety
LANGUAGES = ['en', 'en-us', 'en-uk', 'en-au', 'en-ca']

def add_replay_effect(audio, sr=16000):
    """Simulate replay attack by adding room reverb and quality degradation"""
    # Add reverb (simple echo)
    delay_samples = int(0.05 * sr)  # 50ms delay
    reverb = np.copy(audio)
    if len(audio) > delay_samples:
        reverb[delay_samples:] += audio[:-delay_samples] * 0.3
    
    # Add background noise
    noise = np.random.normal(0, 0.002, len(reverb))
    reverb += noise
    
    # Reduce quality (simulate speaker playback)
    reverb = librosa.resample(reverb, orig_sr=sr, target_sr=8000)
    reverb = librosa.resample(reverb, orig_sr=8000, target_sr=sr)
    
    return reverb

def add_phone_quality(audio, sr=16000):
    """Simulate phone recording quality"""
    # Band-pass filter (phone frequency range)
    audio_filtered = librosa.effects.preemphasis(audio)
    
    # Add compression artifacts
    audio_filtered = np.clip(audio_filtered * 1.2, -0.9, 0.9)
    
    return audio_filtered

# Generate spoofed samples
print("\n🔄 Generating TTS + Effects...")
new_spoofed_count = 0

for i in tqdm(range(TARGET_SPOOFED), desc="Generating spoofed"):
    try:
        # Random text and language
        text = random.choice(PHRASES)
        lang = random.choice(LANGUAGES)
        
        # Generate TTS
        temp_mp3 = TEMP_DIR / f"temp_{i}.mp3"
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(str(temp_mp3))
        
        # Load and convert to 16kHz WAV
        audio, sr = librosa.load(temp_mp3, sr=16000, mono=True)
        
        # Apply random effects
        effect_choice = random.randint(0, 2)
        if effect_choice == 0:
            # Pure TTS
            processed = audio
        elif effect_choice == 1:
            # TTS + Replay effect
            processed = add_replay_effect(audio)
        else:
            # TTS + Phone quality
            processed = add_phone_quality(audio)
        
        # Normalize
        if np.max(np.abs(processed)) > 0:
            processed = processed / np.max(np.abs(processed)) * 0.9
        
        # Save
        output_file = SPOOFED_DIR / f"spoofed_{start_idx + new_spoofed_count:04d}.wav"
        sf.write(output_file, processed, 16000)
        new_spoofed_count += 1
        
        # Clean up temp file
        if temp_mp3.exists():
            temp_mp3.unlink()
        
    except Exception as e:
        print(f"\n⚠️ Error generating sample {i}: {e}")
        continue

print(f"\n✅ Generated {new_spoofed_count} new spoofed samples")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "="*60)
print("✅ DATASET BUILD COMPLETE!")
print("="*60)

# Count final samples
final_genuine = len(list(GENUINE_DIR.glob("genuine_*.wav")))
final_spoofed = len(list(SPOOFED_DIR.glob("spoofed_*.wav")))

print(f"\n📊 Final Dataset Statistics:")
print(f"   Genuine Samples: {final_genuine}")
print(f"   Spoofed Samples: {final_spoofed}")
print(f"   Total Samples:   {final_genuine + final_spoofed}")
print(f"\n📁 Dataset Location: {DATASET_DIR.absolute()}")

# Cleanup
print("\n🧹 Cleaning up temporary files...")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
print("✅ Cleanup complete!")

print("\n🚀 Next step: Train the model")
print("   Run: python train_antispoofing.py --epochs 25 --batch-size 32")
print("="*60)
