#!/usr/bin/env python3
"""
Prepare ASVspoof 2019 LA Dataset for Training

This script converts the ASVspoof 2019 LA dataset format to the format
expected by train_antispoofing.py

ASVspoof Format:
- LA/ASVspoof2019_LA_train/flac/*.flac
- LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt

Expected Format:
- data/antispoofing_dataset/genuine/*.wav
- data/antispoofing_dataset/spoofed/*.wav
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import soundfile as sf


def prepare_asvspoof_dataset(
    asvspoof_dir='data/antispoofing_dataset/LA',
    output_dir='data/antispoofing_dataset',
    max_samples_per_class=None
):
    """
    Prepare ASVspoof 2019 LA dataset for training
    
    Args:
        asvspoof_dir: Path to extracted LA folder
        output_dir: Output directory for organized dataset
        max_samples_per_class: Optional limit on samples per class (for faster training)
    """
    print("=" * 70)
    print("🔧 PREPARING ASVSPOOF 2019 LA DATASET")
    print("=" * 70)
    
    asvspoof_path = Path(asvspoof_dir)
    output_path = Path(output_dir)
    
    # Check if ASVspoof directory exists
    if not asvspoof_path.exists():
        print(f"\n❌ ERROR: ASVspoof directory not found: {asvspoof_dir}")
        print("\n📥 Please download and extract ASVspoof 2019 LA dataset:")
        print("   1. Download from: https://datashare.ed.ac.uk/handle/10283/3336")
        print("   2. Extract LA.zip to: data/antispoofing_dataset/LA/")
        print("\nExpected structure:")
        print("   data/antispoofing_dataset/LA/")
        print("   ├── ASVspoof2019_LA_train/flac/")
        print("   ├── ASVspoof2019_LA_dev/flac/")
        print("   └── ASVspoof2019_LA_cm_protocols/")
        return False
    
    # Paths to protocol files and audio
    protocol_file = asvspoof_path / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.train.trn.txt'
    audio_dir = asvspoof_path / 'ASVspoof2019_LA_train' / 'flac'
    
    # Check if protocol file exists
    if not protocol_file.exists():
        print(f"\n❌ Protocol file not found: {protocol_file}")
        print("   Please ensure ASVspoof dataset is properly extracted.")
        return False
    
    # Check if audio directory exists
    if not audio_dir.exists():
        print(f"\n❌ Audio directory not found: {audio_dir}")
        return False
    
    print(f"\n📁 Reading protocol file: {protocol_file.name}")
    
    # Create output directories
    genuine_dir = output_path / 'genuine'
    spoofed_dir = output_path / 'spoofed'
    genuine_dir.mkdir(parents=True, exist_ok=True)
    spoofed_dir.mkdir(parents=True, exist_ok=True)
    
    # Read protocol file
    # Format: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID LABEL
    # Example: LA_0079 LA_T_1138215 - - bonafide
    # Example: LA_0079 LA_T_1138216 - A07 spoof
    
    genuine_files = []
    spoofed_files = []
    
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                audio_file = parts[1]  # LA_T_1138215
                label = parts[4]  # bonafide or spoof
                
                if label == 'bonafide':
                    genuine_files.append(audio_file)
                elif label == 'spoof':
                    spoofed_files.append(audio_file)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Genuine samples: {len(genuine_files)}")
    print(f"   Spoofed samples: {len(spoofed_files)}")
    print(f"   Total samples: {len(genuine_files) + len(spoofed_files)}")
    
    # Apply sample limit if specified
    if max_samples_per_class:
        print(f"\n⚠️  Limiting to {max_samples_per_class} samples per class for faster training")
        genuine_files = genuine_files[:max_samples_per_class]
        spoofed_files = spoofed_files[:max_samples_per_class]
        print(f"   Using {len(genuine_files)} genuine + {len(spoofed_files)} spoofed samples")
    
    # Convert and copy genuine files
    print(f"\n📦 Processing genuine samples...")
    genuine_copied = 0
    for audio_file in tqdm(genuine_files, desc="Genuine"):
        src_path = audio_dir / f"{audio_file}.flac"
        dst_path = genuine_dir / f"{audio_file}.wav"
        
        if src_path.exists():
            try:
                # Read FLAC and save as WAV
                audio, sr = sf.read(str(src_path))
                sf.write(str(dst_path), audio, sr)
                genuine_copied += 1
            except Exception as e:
                print(f"\n⚠️  Error processing {audio_file}: {e}")
    
    # Convert and copy spoofed files
    print(f"\n📦 Processing spoofed samples...")
    spoofed_copied = 0
    for audio_file in tqdm(spoofed_files, desc="Spoofed"):
        src_path = audio_dir / f"{audio_file}.flac"
        dst_path = spoofed_dir / f"{audio_file}.wav"
        
        if src_path.exists():
            try:
                # Read FLAC and save as WAV
                audio, sr = sf.read(str(src_path))
                sf.write(str(dst_path), audio, sr)
                spoofed_copied += 1
            except Exception as e:
                print(f"\n⚠️  Error processing {audio_file}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Conversion Summary:")
    print(f"   Genuine samples copied: {genuine_copied}")
    print(f"   Spoofed samples copied: {spoofed_copied}")
    print(f"   Total samples: {genuine_copied + spoofed_copied}")
    print(f"\n📁 Output directory: {output_path}")
    print(f"   {genuine_dir}")
    print(f"   {spoofed_dir}")
    print("\n🚀 Ready for training! Run:")
    print(f"   python train_antispoofing.py --dataset {output_dir}")
    print("=" * 70)
    
    return True


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare ASVspoof 2019 LA Dataset')
    parser.add_argument(
        '--input',
        type=str,
        default='data/antispoofing_dataset/LA',
        help='Path to ASVspoof LA directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/antispoofing_dataset',
        help='Output directory'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per class (for faster training/testing)'
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    success = prepare_asvspoof_dataset(
        asvspoof_dir=args.input,
        output_dir=args.output,
        max_samples_per_class=args.max_samples
    )
    
    if not success:
        print("\n❌ Dataset preparation failed!")
        exit(1)


if __name__ == "__main__":
    main()
