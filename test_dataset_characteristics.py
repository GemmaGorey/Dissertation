"""
Quick test to check MERGE dataset song length characteristics

Add this as a new cell in Dissertation_GG.ipynb to verify:
1. How many songs need padding (too short)
2. How many songs need truncation (too long)
3. Whether the padding/truncation issues actually affect your dataset

Run this AFTER loading final_df but BEFORE processing all songs.
"""

import librosa
import numpy as np
import os
from tqdm.auto import tqdm

# Configuration
base_path = '/content/drive/MyDrive/dissertation/MERGE_Bimodal_Complete'
TARGET_SR = 22050
target_length = 1292  # 30 seconds at sr=22050, hop_length=512

# Sample size (set to None to check all songs, or a number like 100 for quick test)
SAMPLE_SIZE = 100  # Change to None for full dataset check

# Statistics
songs_exact = 0
songs_too_short = 0
songs_too_long = 0
short_lengths = []
long_lengths = []

# Sample or use full dataset
if SAMPLE_SIZE:
    test_df = final_df.sample(n=min(SAMPLE_SIZE, len(final_df)), random_state=42)
    print(f"Testing {len(test_df)} randomly sampled songs...\n")
else:
    test_df = final_df
    print(f"Testing ALL {len(test_df)} songs...\n")

# Check each song
for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Checking songs"):
    song_id = row['Audio_Song']
    quadrant = row['Quadrant']
    audio_path = os.path.join(base_path, 'audio', quadrant, f"{song_id}.mp3")

    try:
        # Load audio
        audio_waveform, sample_rate = librosa.load(audio_path, sr=TARGET_SR)

        # Create mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_waveform,
            sr=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        current_length = mel_spectrogram.shape[1]

        # Categorize
        if current_length == target_length:
            songs_exact += 1
        elif current_length < target_length:
            songs_too_short += 1
            short_lengths.append(current_length)
        else:  # current_length > target_length
            songs_too_long += 1
            long_lengths.append(current_length)

    except Exception as e:
        print(f"Error processing {song_id}: {e}")

# Results
print("\n" + "="*70)
print("DATASET CHARACTERISTICS REPORT")
print("="*70)
print(f"\nTarget length: {target_length} frames (≈30 seconds)")
print(f"Total songs checked: {len(test_df)}")
print(f"\n📊 LENGTH DISTRIBUTION:")
print(f"  - Exact length (no padding/truncation needed): {songs_exact} ({songs_exact/len(test_df)*100:.1f}%)")
print(f"  - Too short (will be PADDED):                 {songs_too_short} ({songs_too_short/len(test_df)*100:.1f}%)")
print(f"  - Too long (will be TRUNCATED):               {songs_too_long} ({songs_too_long/len(test_df)*100:.1f}%)")

if songs_too_short > 0:
    print(f"\n⚠️ PADDING ISSUE AFFECTS {songs_too_short} SONGS")
    print(f"   Shortest song: {min(short_lengths)} frames ({min(short_lengths)/1292*30:.1f} seconds)")
    print(f"   Average short song: {np.mean(short_lengths):.0f} frames ({np.mean(short_lengths)/1292*30:.1f} seconds)")
    print(f"   Padding needed: {target_length - min(short_lengths)} frames max")
    print(f"   → Issue 3 (padding bug) IS RELEVANT ❌")
else:
    print(f"\n✅ NO PADDING NEEDED - Issue 3 not relevant")

if songs_too_long > 0:
    print(f"\n⚠️ TRUNCATION AFFECTS {songs_too_long} SONGS")
    print(f"   Longest song: {max(long_lengths)} frames ({max(long_lengths)/1292*30:.1f} seconds)")
    print(f"   Average long song: {np.mean(long_lengths):.0f} frames ({np.mean(long_lengths)/1292*30:.1f} seconds)")
    print(f"   → Issue 5 (truncation strategy) IS RELEVANT ❌")
else:
    print(f"\n✅ NO TRUNCATION NEEDED - Issue 5 not relevant")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)

if songs_too_short > 0 or songs_too_long > 0:
    print("❌ Your dataset HAS songs that need padding/truncation")
    print("   → Issue 3 (padding) should be FIXED")
    if songs_too_long > 0:
        print("   → Issue 5 (truncation) should be FIXED")
else:
    print("✅ Your dataset is perfect - all songs are exactly 30 seconds")
    print("   → Issue 3 and 5 are not relevant")

print("\nNext steps:")
if songs_too_short > 0:
    print("1. Fix padding: change to constant_values=-80")
if songs_too_long > 0:
    print("2. Fix truncation: use middle segment instead of first")
print("3. Reprocess spectrograms with fixed code")
print("="*70)
