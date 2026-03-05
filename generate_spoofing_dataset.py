#!/usr/bin/env python3
"""
Anti-Spoofing Dataset Generator

Generates a training dataset for anti-spoofing model using:
1. Genuine samples from enrolled users
2. Synthetic spoofed samples using TTS and audio manipulation
"""

import os
import sys
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from gtts import gTTS
import tempfile
from scipy import signal
from tqdm import tqdm

class SpoofingDatasetGenerator:
    """Generate anti-spoofing training dataset"""
    
    def __init__(self, output_dir='data/antispoofing_dataset'):
        self.output_dir = Path(output_dir)
        self.genuine_dir = self.output_dir / 'genuine'
        self.spoofed_dir = self.output_dir / 'spoofed'
        
        # Create directories
        self.genuine_dir.mkdir(parents=True, exist_ok=True)
        self.spoofed_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample phrases for TTS generation
        self.phrases = [
            "My voice is my password verify me",
            "Authentication is successful",
            "Access granted to the system",
            "Welcome to the voice authentication",
            "Please verify my identity",
            "Confirm my access to the application",
            "I am the authorized user",
            "Grant me entry to the secure area",
            "Voice biometric authentication",
            "Secure access verification",
            "Identity confirmation required",
            "Biometric login process",
            "Voice recognition system",
            "User authentication protocol",
            "Security verification complete",
            "Access control verification",
            "Voice signature validation",
            "Authorized user login",
            "Biometric security check",
            "Voice pattern recognition"
        ]
    
    def collect_genuine_samples(self, source_dir='data/voice_profiles'):
        """
        Collect genuine voice samples from enrolled users
        
        Args:
            source_dir: Directory containing voice profiles
        """
        print("📁 Collecting genuine voice samples...")
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"⚠️  Voice profiles directory not found: {source_dir}")
            print("   Creating sample genuine audio from silence...")
            return 0
        
        genuine_count = 0
        
        # Search for .npy files (voice embeddings stored as audio)
        for profile_file in source_path.rglob('*.npy'):
            try:
                # Load the numpy array
                audio_data = np.load(profile_file)
                
                # If it's embeddings, skip it
                if len(audio_data.shape) == 1 and len(audio_data) < 1000:
                    continue
                
                # Save as audio file
                output_file = self.genuine_dir / f"genuine_{genuine_count:04d}.wav"
                sf.write(output_file, audio_data, 16000)
                genuine_count += 1
                
            except Exception as e:
                continue
        
        # Also look for WAV files
        for wav_file in source_path.rglob('*.wav'):
            try:
                audio, sr = librosa.load(wav_file, sr=16000)
                
                # Skip if too short
                if len(audio) < 16000:  # Less than 1 second
                    continue
                
                output_file = self.genuine_dir / f"genuine_{genuine_count:04d}.wav"
                sf.write(output_file, audio, 16000)
                genuine_count += 1
                
            except Exception as e:
                continue
        
        print(f"✅ Collected {genuine_count} genuine samples")
        return genuine_count
    
    def generate_tts_spoofs(self, num_samples=200):
        """
        Generate TTS-based spoofed samples
        
        Args:
            num_samples: Number of TTS samples to generate
        """
        print(f"\n🤖 Generating {num_samples} TTS spoofed samples...")
        
        languages = ['en', 'en-us', 'en-uk', 'en-au']
        
        for i in tqdm(range(num_samples), desc="TTS Generation"):
            try:
                # Select random phrase
                phrase = random.choice(self.phrases)
                lang = random.choice(languages)
                
                # Generate TTS
                tts = gTTS(text=phrase, lang=lang, slow=False)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_path = temp_file.name
                    tts.save(temp_path)
                
                # Load and convert to 16kHz WAV
                audio, sr = librosa.load(temp_path, sr=16000)
                
                # Remove silence
                audio, _ = librosa.effects.trim(audio, top_db=20)
                
                # Save
                output_file = self.spoofed_dir / f"tts_{i:04d}.wav"
                sf.write(output_file, audio, 16000)
                
                # Clean up temp file
                os.unlink(temp_path)
                
            except Exception as e:
                print(f"   Error generating TTS sample {i}: {e}")
                continue
        
        print(f"✅ Generated TTS spoofed samples")
    
    def generate_replay_spoofs(self, num_samples=100):
        """
        Simulate replay attacks by adding artifacts
        
        Args:
            num_samples: Number of replay spoofs to generate
        """
        print(f"\n🔊 Generating {num_samples} replay attack simulations...")
        
        # Get genuine samples to create replays from
        genuine_files = list(self.genuine_dir.glob('*.wav'))
        
        if len(genuine_files) == 0:
            print("   ⚠️  No genuine samples available for replay simulation")
            return
        
        for i in tqdm(range(num_samples), desc="Replay Simulation"):
            try:
                # Pick random genuine sample
                source_file = random.choice(genuine_files)
                audio, sr = librosa.load(source_file, sr=16000)
                
                # Simulate replay artifacts
                
                # 1. Add room reverb
                audio = self._add_reverb(audio, sr)
                
                # 2. Apply lowpass filter (simulate speaker frequency response)
                audio = self._apply_lowpass(audio, sr, cutoff=random.uniform(3000, 6000))
                
                # 3. Add playback noise
                audio = self._add_playback_noise(audio)
                
                # 4. Reduce high frequencies
                audio = self._reduce_high_freq(audio, sr)
                
                # Normalize
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
                
                # Save
                output_file = self.spoofed_dir / f"replay_{i:04d}.wav"
                sf.write(output_file, audio, 16000)
                
            except Exception as e:
                print(f"   Error generating replay sample {i}: {e}")
                continue
        
        print(f"✅ Generated replay spoofed samples")
    
    def generate_processed_spoofs(self, num_samples=100):
        """
        Generate spoofs with heavy processing/effects
        
        Args:
            num_samples: Number of processed spoofs to generate
        """
        print(f"\n⚙️  Generating {num_samples} processed/manipulated spoofs...")
        
        genuine_files = list(self.genuine_dir.glob('*.wav'))
        
        if len(genuine_files) == 0:
            print("   ⚠️  No genuine samples available")
            return
        
        for i in tqdm(range(num_samples), desc="Processing"):
            try:
                source_file = random.choice(genuine_files)
                audio, sr = librosa.load(source_file, sr=16000)
                
                # Apply random processing
                processing_type = random.choice(['pitch', 'speed', 'compress', 'distort'])
                
                if processing_type == 'pitch':
                    # Pitch shift
                    n_steps = random.uniform(-4, 4)
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
                
                elif processing_type == 'speed':
                    # Time stretch
                    rate = random.uniform(0.8, 1.2)
                    audio = librosa.effects.time_stretch(audio, rate=rate)
                
                elif processing_type == 'compress':
                    # Heavy compression simulation
                    audio = self._add_compression(audio)
                
                elif processing_type == 'distort':
                    # Add distortion
                    audio = np.clip(audio * random.uniform(1.5, 3.0), -1.0, 1.0)
                
                # Normalize
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
                
                output_file = self.spoofed_dir / f"processed_{i:04d}.wav"
                sf.write(output_file, audio, 16000)
                
            except Exception as e:
                print(f"   Error generating processed sample {i}: {e}")
                continue
        
        print(f"✅ Generated processed spoofed samples")
    
    def _add_reverb(self, audio, sr, room_scale=0.3):
        """Add simple reverb effect"""
        delay_samples = int(sr * 0.05)  # 50ms delay
        reverb = np.zeros_like(audio)
        
        for i in range(5):
            delay = delay_samples * (i + 1)
            decay = 0.6 ** (i + 1)
            if delay < len(audio):
                reverb[delay:] += audio[:-delay] * decay
        
        return audio + reverb * room_scale
    
    def _apply_lowpass(self, audio, sr, cutoff=4000):
        """Apply lowpass filter"""
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio)
    
    def _add_playback_noise(self, audio, noise_level=0.005):
        """Add low-level noise simulating playback"""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def _reduce_high_freq(self, audio, sr):
        """Reduce high frequency content"""
        cutoff = random.uniform(6000, 8000)
        return self._apply_lowpass(audio, sr, cutoff)
    
    def _add_compression(self, audio, threshold=0.3, ratio=4):
        """Simple compression effect"""
        compressed = audio.copy()
        mask = np.abs(audio) > threshold
        compressed[mask] = threshold + (audio[mask] - threshold) / ratio
        return compressed
    
    def generate_labels_file(self):
        """Generate protocol/labels file"""
        print("\n📝 Generating labels file...")
        
        labels_file = self.output_dir / 'labels.txt'
        
        with open(labels_file, 'w') as f:
            f.write("filename,label\n")
            
            # Genuine samples
            for file in self.genuine_dir.glob('*.wav'):
                f.write(f"{file.name},genuine\n")
            
            # Spoofed samples
            for file in self.spoofed_dir.glob('*.wav'):
                f.write(f"{file.name},spoof\n")
        
        print(f"✅ Labels saved to {labels_file}")
    
    def generate_dataset(self, num_tts=200, num_replay=100, num_processed=100):
        """
        Generate complete anti-spoofing dataset
        
        Args:
            num_tts: Number of TTS samples
            num_replay: Number of replay samples
            num_processed: Number of processed samples
        """
        print("=" * 60)
        print("🎯 ANTI-SPOOFING DATASET GENERATOR")
        print("=" * 60)
        
        # Step 1: Collect genuine samples
        genuine_count = self.collect_genuine_samples()
        
        # Step 2: Generate TTS spoofs
        self.generate_tts_spoofs(num_tts)
        
        # Step 3: Generate replay simulations (only if we have genuine samples)
        if genuine_count > 0:
            self.generate_replay_spoofs(num_replay)
            self.generate_processed_spoofs(num_processed)
        else:
            print("\n⚠️  No genuine samples found, skipping replay/processed generation")
        
        # Step 4: Generate labels
        self.generate_labels_file()
        
        # Summary
        total_genuine = len(list(self.genuine_dir.glob('*.wav')))
        total_spoofed = len(list(self.spoofed_dir.glob('*.wav')))
        
        print("\n" + "=" * 60)
        print("✅ DATASET GENERATION COMPLETE!")
        print("=" * 60)
        print(f"📊 Total genuine samples: {total_genuine}")
        print(f"📊 Total spoofed samples: {total_spoofed}")
        print(f"📊 Total samples: {total_genuine + total_spoofed}")
        print(f"📁 Dataset location: {self.output_dir.absolute()}")
        print("\n🚀 Ready for training!")
        print("   Run: python train_antispoofing.py")
        print("=" * 60)


def main():
    """Main execution"""
    generator = SpoofingDatasetGenerator()
    
    # Generate dataset
    # Adjust numbers based on your needs (lower for faster generation)
    generator.generate_dataset(
        num_tts=200,          # TTS samples (takes ~10-15 min)
        num_replay=100,       # Replay simulations
        num_processed=100     # Processed samples
    )


if __name__ == "__main__":
    main()
