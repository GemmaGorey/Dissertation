"""
Diagnostic Script: Find Problem Songs in MERGE Dataset

This script will:
1. Find the shortest and longest songs
2. Log all songs that aren't exactly 30 seconds
3. Help investigate why songs aren't the expected length
4. Save a CSV of problem songs for manual inspection

Add this as a new cell in Dissertation_GG.ipynb
"""

import librosa
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm

# Configuration
base_path = '/content/drive/MyDrive/dissertation/MERGE_Bimodal_Complete'
TARGET_SR = 22050
target_length = 1292  # 30 seconds
hop_length = 512

# Storage for problem songs
problem_songs = []

print("🔍 Finding problem songs in MERGE dataset...")
print("="*70)

# Check ALL songs
for index, row in tqdm(final_df.iterrows(), total=len(final_df), desc="Analyzing songs"):
    song_id = row['Audio_Song']
    quadrant = row['Quadrant']
    audio_path = os.path.join(base_path, 'audio', quadrant, f"{song_id}.mp3")

    try:
        # Load audio
        audio_waveform, sample_rate = librosa.load(audio_path, sr=TARGET_SR)

        # Get actual duration
        duration_seconds = len(audio_waveform) / sample_rate

        # Create mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_waveform,
            sr=sample_rate,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=128
        )

        current_length = mel_spectrogram.shape[1]

        # Calculate how far from target
        difference = current_length - target_length
        difference_seconds = (current_length - target_length) / (sample_rate / hop_length)

        # Store if not exact length
        if current_length != target_length:
            problem_songs.append({
                'song_id': song_id,
                'quadrant': quadrant,
                'path': audio_path,
                'spectrogram_frames': current_length,
                'target_frames': target_length,
                'difference_frames': difference,
                'actual_duration_seconds': duration_seconds,
                'expected_duration_seconds': 30.0,
                'difference_seconds': duration_seconds - 30.0,
                'status': 'TOO_SHORT' if current_length < target_length else 'TOO_LONG'
            })

    except Exception as e:
        print(f"❌ Error processing {song_id}: {e}")
        problem_songs.append({
            'song_id': song_id,
            'quadrant': quadrant,
            'path': audio_path,
            'status': 'ERROR',
            'error': str(e)
        })

# Create DataFrame
problem_df = pd.DataFrame(problem_songs)

# Sort by duration to find extremes
problem_df_sorted = problem_df[problem_df['status'] != 'ERROR'].sort_values('actual_duration_seconds')

print("\n" + "="*70)
print("🔍 PROBLEM SONGS ANALYSIS")
print("="*70)

# Show shortest songs
print("\n📉 TOP 10 SHORTEST SONGS:")
print("-"*70)
shortest = problem_df_sorted.head(10)
for idx, row in shortest.iterrows():
    print(f"  {row['song_id']:<20} | {row['actual_duration_seconds']:>6.2f}s | {row['quadrant']}")
    print(f"    → Path: {row['path']}")
    print(f"    → Frames: {row['spectrogram_frames']} (need {row['difference_frames']*-1:.0f} more)")
    print()

# Show longest songs
print("\n📈 TOP 10 LONGEST SONGS:")
print("-"*70)
longest = problem_df_sorted.tail(10)
for idx, row in longest.iterrows():
    print(f"  {row['song_id']:<20} | {row['actual_duration_seconds']:>6.2f}s | {row['quadrant']}")
    print(f"    → Path: {row['path']}")
    print(f"    → Frames: {row['spectrogram_frames']} (extra {row['difference_frames']:.0f})")
    print()

# Statistics
too_short = problem_df[problem_df['status'] == 'TOO_SHORT']
too_long = problem_df[problem_df['status'] == 'TOO_LONG']

print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)
print(f"\nTotal songs analyzed: {len(final_df)}")
print(f"Problem songs found: {len(problem_df)}")
print(f"  - Too short: {len(too_short)} ({len(too_short)/len(final_df)*100:.1f}%)")
print(f"  - Too long: {len(too_long)} ({len(too_long)/len(final_df)*100:.1f}%)")

if len(too_short) > 0:
    print(f"\nToo Short Songs:")
    print(f"  - Shortest: {too_short['actual_duration_seconds'].min():.2f}s")
    print(f"  - Average: {too_short['actual_duration_seconds'].mean():.2f}s")
    print(f"  - Most are around: {too_short['actual_duration_seconds'].median():.2f}s")

if len(too_long) > 0:
    print(f"\nToo Long Songs:")
    print(f"  - Longest: {too_long['actual_duration_seconds'].max():.2f}s")
    print(f"  - Average: {too_long['actual_duration_seconds'].mean():.2f}s")
    print(f"  - Most are around: {too_long['actual_duration_seconds'].median():.2f}s")

# Save to CSV for manual inspection
output_path = '/content/drive/MyDrive/dissertation/problem_songs_analysis.csv'
problem_df.to_csv(output_path, index=False)
print(f"\n💾 Full report saved to: {output_path}")

print("\n" + "="*70)
print("🤔 POSSIBLE EXPLANATIONS")
print("="*70)
print("""
Why songs aren't exactly 30 seconds:

1. MERGE dataset preprocessing variations
   - Songs may have been trimmed slightly differently
   - Silence removal at start/end
   - Compression artifacts

2. The 4.8 second song specifically:
   - Could be corrupted file
   - Could be wrong file in dataset
   - Could be metadata error
   - Should manually inspect this file!

3. The "too long" songs (30.1-30.8s):
   - Very close to 30s (only 0.1-0.8s over)
   - Probably due to rounding in original preprocessing
   - Less concerning than the very short ones

RECOMMENDATIONS:
✅ Fix padding/truncation as planned (this data confirms it's needed)
⚠️ Manually check the shortest songs - especially the 4.8s one
⚠️ Consider excluding extremely short songs (<20s?) as potentially corrupted
""")

print("\n" + "="*70)
print("🔎 TO INSPECT THE 4.8 SECOND SONG:")
print("="*70)
shortest_song = problem_df_sorted.iloc[0]
print(f"\nSong ID: {shortest_song['song_id']}")
print(f"Quadrant: {shortest_song['quadrant']}")
print(f"Path: {shortest_song['path']}")
print(f"Duration: {shortest_song['actual_duration_seconds']:.2f} seconds")
print(f"\nNext steps:")
print("1. Listen to this file to verify it's actually that short")
print("2. Check if it's corrupted")
print("3. Check MERGE dataset documentation for this song")
print("4. Consider excluding if it's an error")
print("="*70)
