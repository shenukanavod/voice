#!/usr/bin/env python3
"""
Download Genuine Voice Samples

Downloads a small subset of genuine voice samples from LibriSpeech dataset
for anti-spoofing model training.
"""

import os
import sys
import urllib.request
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf


class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_librispeech_subset(output_dir='data/antispoofing_dataset/genuine'):
    """
    Download LibriSpeech dev-clean subset (small, clean speech)
    
    Args:
        output_dir: Where to save genuine samples
    """
    print("=" * 60)
    print("📥 DOWNLOADING GENUINE VOICE SAMPLES")
    print("=" * 60)
    print("\nDataset: LibriSpeech dev-clean (small subset)")
    print("Size: ~350 MB")
    print("Samples: ~2,700 clean speech samples")
    print()
    
    # Create directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path('data/temp_download')
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Download URL
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = temp_dir / "dev-clean.tar.gz"
    
    try:
        # Download
        print("⬇️  Downloading LibriSpeech dev-clean...")
        download_file(url, tar_path)
        print(f"✅ Download complete: {tar_path}")
        
        # Extract
        print("\n📦 Extracting archive...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(temp_dir)
        print("✅ Extraction complete")
        
        # Process and copy samples
        print("\n🔄 Processing audio files...")
        
        librispeech_dir = temp_dir / "LibriSpeech" / "dev-clean"
        sample_count = 0
        max_samples = 300  # Limit to 300 samples for faster training
        
        # Find all FLAC files
        flac_files = list(librispeech_dir.rglob('*.flac'))
        print(f"   Found {len(flac_files)} FLAC files")
        print(f"   Converting first {max_samples} to WAV format...")
        
        for flac_file in tqdm(flac_files[:max_samples], desc="Converting"):
            try:
                # Load audio
                audio, sr = librosa.load(flac_file, sr=16000)
                
                # Skip if too short
                if len(audio) < 16000:  # Less than 1 second
                    continue
                
                # Trim to max 5 seconds
                if len(audio) > 16000 * 5:
                    audio = audio[:16000 * 5]
                
                # Save as WAV
                output_file = output_path / f"genuine_{sample_count:04d}.wav"
                sf.write(output_file, audio, 16000)
                
                sample_count += 1
                
            except Exception as e:
                print(f"\n   Error processing {flac_file.name}: {e}")
                continue
        
        print(f"\n✅ Processed {sample_count} genuine voice samples")
        
        # Cleanup
        print("\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✅ Cleanup complete")
        
        print("\n" + "=" * 60)
        print("✅ GENUINE SAMPLES DOWNLOAD COMPLETE!")
        print("=" * 60)
        print(f"📊 Total samples: {sample_count}")
        print(f"📁 Location: {output_path.absolute()}")
        print("\n🚀 Next step: Re-run dataset generator")
        print("   Run: python generate_spoofing_dataset.py")
        print("=" * 60)
        
        return sample_count
        
    except Exception as e:
        print(f"\n❌ Error downloading samples: {e}")
        print("\n💡 Alternative: Enroll a few users in the UI first,")
        print("   then run the dataset generator again.")
        return 0


def main():
    """Main execution"""
    download_librispeech_subset()


if __name__ == "__main__":
    main()
