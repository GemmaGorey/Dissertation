# Implementation Guide: Critical Fixes and SE-Attention Upgrade

**Date:** 2025-10-30
**Purpose:** Complete code fixes for preprocessing bugs and SE-Attention implementation

---

## 📋 TODO LIST

### 🎯 QUICK SUMMARY - WHAT'S LEFT TO DO:

**PHASE 1: ✅ COMPLETE** - All preprocessing fixes done, all 2216 spectrograms reprocessed

**PHASE 2: ⏳ TODO** - Update model architectures (MODEL_3, 4, 5) with SE-Attention
- Replace attention mechanism in 3 notebooks
- Estimated time: 30-45 minutes

**PHASE 3: ⏳ TODO** - Add training improvements (gradient clipping, epoch tracking)
- Small edits to training loops in 3 notebooks
- Estimated time: 15-20 minutes

**PHASE 4: ⏳ TODO** - Retrain all models and evaluate improvements
- Run training for MODEL_3, 4, 5 with fixed data
- Estimated time: Several hours (training time)

---

### ✅ PHASE 1: CRITICAL PREPROCESSING FIXES - **COMPLETED!**

- [x] **Task 1.1:** ✅ Created `Audio_Processing_V2_Dissertation_GG.ipynb`
- [x] **Task 1.2:** ✅ Fixed `load_song_data()` in cell 16
- [x] **Task 1.3:** ✅ Decided on normalization: `ref=GLOBAL_MAX` (dataset-wide consistency)
- [x] **Task 1.4:** ✅ Verified fix - spectrograms checked with cell 17
- [x] **Task 1.5:** ✅ Reprocessed ALL 2216 spectrograms (cell 18 completed)
- [x] **Task 1.6:** ✅ Visually verified ALL spectrograms using cell 20 batch display
- [x] **Task 1.7:** ✅ Confirmed all spectrograms look correct

**Status:** ALL PREPROCESSING FIXES COMPLETE ✅

**What was implemented (Audio_Processing_V2_Dissertation_GG.ipynb):**
- ✅ **Cell 16, line 29:** `ref=GLOBAL_MAX` in power_to_db (dataset-wide normalization consistency)
  - GLOBAL_MAX = 13400.846 (computed in cell 13)
  - Ensures all spectrograms normalized to same reference
  - Preserves relative loudness across entire dataset
- ✅ **Cell 16, line 40:** `constant_values=-80.0` for padding (silence, not artificial loudness)
- ✅ **Cell 16, line 35:** First segment truncation (appropriate for ~30s songs)
- ✅ **Cell 18:** All 2216 songs processed and saved successfully
- ✅ **Cell 19:** Master CSV verified with correct file paths
- ✅ **Cell 20:** NEW - Batch visualization of ALL spectrograms (24 per batch, 93 batches total)
  - Displays all 2216 spectrograms in 6×4 grids
  - User confirmed: **all spectrograms look OK!**

### 🟡 PHASE 2: MOVE ATTENTION EARLIER (OPTIONAL BUT RECOMMENDED)

**What:** Move existing AttentionModule earlier in the network (before final classifier)

**Why:** Attention will learn which of the 512 CNN features are important (richer representation) instead of which of the 64 final features are important (compressed representation)

**Tasks:**
- [ ] **Task 2.1:** Update `VGGish_Audio_Model` in MODEL_3 (cell-6)
  - Change `AttentionModule(64)` → `AttentionModule(512)`
  - Move `self.attention(x)` to BEFORE classifier in forward()
- [ ] **Task 2.2:** Update `VGGish_Audio_Model` in MODEL_4 (cell-6)
  - Same changes (keep 2 conv layers per block)
- [ ] **Task 2.3:** Update `VGGish_Audio_Model` in MODEL_5 (cell-6)
  - Same changes
- [ ] **Task 2.4:** Verify model loads with test code

**Estimated time:** 15-20 minutes (2 simple changes per notebook × 3 notebooks)

**What stays the same:**
- ✅ Cell-5 (AttentionModule) - NO CHANGES NEEDED!
- ✅ Your CNN architecture
- ✅ Your classifier structure

### 🟢 PHASE 3: TRAINING IMPROVEMENTS (DO THIRD)

- [ ] **Task 3.1:** Add gradient clipping to MODEL_3, 4, 5 (cell-10)
- [ ] **Task 3.2:** Fix epoch tracking in MODEL_3, 4, 5 (cell-10)
- [ ] **Task 3.3:** Test training loop runs without errors

**Estimated time:** 15 minutes

### 🚀 PHASE 4: RETRAIN AND EVALUATE

- [ ] **Task 4.1:** Retrain MODEL_3 with all fixes
- [ ] **Task 4.2:** Retrain MODEL_4 with all fixes
- [ ] **Task 4.3:** Retrain MODEL_5 with all fixes
- [ ] **Task 4.4:** Compare old vs new results
- [ ] **Task 4.5:** Document improvements in dissertation

**Estimated time:** Several hours (training time)

---

## 🔴 PHASE 1: PREPROCESSING FIXES

### File to Edit: `Dissertation_GG.ipynb`
### Cell to Replace: **cell-10** (the `load_song_data` function)

**BEFORE (current buggy version):**
```python
def load_song_data(song_id, lyric_id, quadrant):
    # ... existing code ...

    # Convert to decibels
    db_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # ❌ BUG 1

    target_length = 1292
    current_length = db_spectrogram.shape[1]

    if current_length > target_length:
        db_spectrogram = db_spectrogram[:, :target_length]  # ❌ BUG 2
    elif current_length < target_length:
        padding_needed = target_length - current_length
        db_spectrogram = np.pad(db_spectrogram, ((0, 0), (0, padding_needed)), mode='constant')  # ❌ BUG 3

    # ... rest of function ...
```

**AFTER (fixed version) - COPY THIS ENTIRE FUNCTION:**

```python
def load_song_data(song_id, lyric_id, quadrant):
    """
    Loads the audio and lyrics for a given song ID
    FIXED VERSION with corrected normalization, padding, and truncation
    """
    print(f"Attempting to load song: {song_id}")
    try:
        # Construct the file path to google drive
        base_path = '/content/drive/MyDrive/dissertation/MERGE_Bimodal_Complete'

        # Audio files url addition as these are in subfolders for each emotion quadrant
        audio_path = os.path.join(base_path, 'audio', quadrant, f"{song_id}.mp3")

        # Lyric files url addition as these are in a separate main folder
        lyrics_path = os.path.join(base_path, 'lyrics', quadrant, f"{lyric_id}.txt")

        # load the audio file
        TARGET_SR = 22050
        audio_waveform, sample_rate = librosa.load(audio_path, sr=TARGET_SR)

        # Process Audio into a Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_waveform,
            sr=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        # ✅ FIX 1: Convert to decibels with CONSISTENT reference
        # Using ref=1.0 ensures all spectrograms are normalized the same way
        # This preserves absolute loudness information (important for arousal)
        db_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0)

        target_length = 1292  # 30 seconds at sr=22050, hop_length=512
        current_length = db_spectrogram.shape[1]

        if current_length > target_length:
            # ✅ FIX 2: Use middle segment instead of first 30 seconds
            # This is more likely to capture important emotional moments (chorus, etc.)
            start_frame = (current_length - target_length) // 2
            db_spectrogram = db_spectrogram[:, start_frame:start_frame + target_length]

        elif current_length < target_length:
            # ✅ FIX 3: Pad with minimum value instead of 0
            # After power_to_db, values are in dB (can be very negative)
            # Padding with 0 would create artificial "loud" regions
            # We pad with -80 dB (very quiet) which is more appropriate
            padding_needed = target_length - current_length
            db_spectrogram = np.pad(
                db_spectrogram,
                ((0, 0), (0, padding_needed)),
                mode='constant',
                constant_values=-80  # -80 dB is essentially silence
            )

        # load lyrics text
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            lyrics_text = f.read()

        # Tokenise raw text
        encoded_lyrics = tokenizer(
            lyrics_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        print(f"Successfully loaded and processed {song_id}")
        return db_spectrogram, encoded_lyrics

    except Exception as e:
        print(f"⚠️ An error occurred for {song_id}: {e}")
        return None, None
```

### 📝 What Changed?

1. **Line ~40:** `ref=np.max` → `ref=1.0` (consistent normalization)
2. **Line ~48:** Takes middle segment instead of first segment
3. **Line ~56:** Pads with `-80` instead of `0` (silence, not loud)
4. **Line ~85:** Added warning emoji to error message (optional, for visibility)

### ✅ Testing the Fix

After replacing cell-10, run this test in cell-11 (should already exist):

```python
# Testing the fixed function
test_song_index = 1111

test_audio_id = final_df.iloc[test_song_index]['Audio_Song']
test_lyric_id = final_df.iloc[test_song_index]['Lyric_Song_x']
test_quadrant = final_df.iloc[test_song_index]['Quadrant']

print(f"Testing fixed preprocessing on song: {test_audio_id}")
spectrogram, encoded_lyrics = load_song_data(test_audio_id, test_lyric_id, test_quadrant)

if spectrogram is not None:
    print(f"\n✅ Spectrogram shape: {spectrogram.shape}")
    print(f"✅ Spectrogram min: {spectrogram.min():.2f} dB")
    print(f"✅ Spectrogram max: {spectrogram.max():.2f} dB")
    print(f"✅ Spectrogram mean: {spectrogram.mean():.2f} dB")

    # Verify the last frames (should be -80 if padded, not 0)
    if spectrogram.shape[1] == 1292:
        last_column_mean = spectrogram[:, -1].mean()
        print(f"✅ Last column mean: {last_column_mean:.2f} dB (should be ~-80 if padded, not 0)")
```

**Expected output:**
- Min should be around -80 dB (not lower)
- Max should be reasonable (not 0 in padded regions)
- Last column should be -80 if song was padded

---

## 🟡 PHASE 2: MOVE ATTENTION EARLIER IN THE NETWORK

### Files to Edit: MODEL_3, MODEL_4, MODEL_5 notebooks

**What we're doing:** Moving your existing AttentionModule to operate on 512-dimensional features (after CNN, before classifier) instead of 64-dimensional features (after classifier).

**Why:** Attention will learn which of the 512 CNN features are important, rather than which of the 64 final features. This gives the network more capacity to focus on important patterns.

---

### OPTION A: Simple Placement Change (RECOMMENDED)

This keeps your existing `AttentionModule` unchanged, just moves where it's applied.

### STEP 1: Keep cell-5 - AttentionModule stays the same!

**File:** `MODEL_3_Dissertation_model_training_basic_VGGish.ipynb`
**Cell:** cell-5

**NO CHANGES NEEDED** - Your AttentionModule is fine as-is:

```python
class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        '''
        Attention mechanism to weight the importance of different features
        '''
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),  # e.g., 512 → 128
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),  # e.g., 128 → 512
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, feature_dim]
        attention_weights = self.attention(x)  # [batch_size, feature_dim]
        weighted_features = x * attention_weights  # Element-wise multiplication
        return weighted_features
```

✅ This code works for any `feature_dim` (64, 512, etc.), so no changes needed!

---

### STEP 2: Update cell-6 - VGGish_Audio_Model (Move Attention Earlier)

**File:** `MODEL_3_Dissertation_model_training_basic_VGGish.ipynb`
**Cell:** cell-6

**CURRENT CODE (attention applied on 64-dim features):**
```python
class VGGish_Audio_Model(nn.Module):
    def __init__(self):
        super(VGGish_Audio_Model, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64)
        )

        self.attention = AttentionModule(64)  # ❌ Operates on 64 dims

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)       # [B, 512] → [B, 64]
        x = self.attention(x)        # ❌ Attention on 64 dims (after classifier)
        return x
```

**NEW CODE (attention applied on 512-dim features - COPY THIS):**

```python
class VGGish_Audio_Model(nn.Module):
    '''VGG-style model with attention moved earlier
      - Attention now operates on 512-dim features (after CNN pooling, before classifier)
      - This allows attention to learn feature importance at a richer representation level
    '''

    def __init__(self):
        super(VGGish_Audio_Model, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Pool to [B, 512, 1, 1]
        )

        # ✅ MOVED: Attention now operates on 512 dimensions
        self.attention = AttentionModule(512)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64)  # Final feature vector size should be 64
        )

    def forward(self, x):
        # CNN feature extraction: [B, 1, 128, 1292] → [B, 512, 1, 1]
        x = self.features(x)

        # Flatten for attention: [B, 512, 1, 1] → [B, 512]
        x = x.view(x.size(0), -1)

        # ✅ MOVED: Apply attention on 512-dim features (before classifier)
        x = self.attention(x)  # [B, 512] → [B, 512] with attention weights

        # Classifier: [B, 512] → [B, 64]
        x = self.classifier(x)

        return x
```

**Key changes:**
1. Line 38: Changed `AttentionModule(64)` → `AttentionModule(512)`
2. Line 52: Moved `self.attention(x)` to BEFORE classifier (not after)

**Repeat this same change for MODEL_4 and MODEL_5**

---

### STEP 3: Special Instructions for MODEL_4

**MODEL_4 has 2 conv layers per block** (more like true VGG). Make the same changes but keep the extra convolutions:

**Changes for MODEL_4:**
1. Change `self.attention = AttentionModule(64)` → `AttentionModule(512)`
2. Move `self.attention(x)` to BEFORE classifier in the forward() method

Everything else is the same as MODEL_3, just keep your 2 convolutions per block.

---

### STEP 4: Verify the Changes Work

After updating all three notebooks, test that the model loads:

```python
# Add this test cell after your model definition
model = BimodalClassifier()
print("Model created successfully!")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test a forward pass
import torch
dummy_spec = torch.randn(2, 1, 128, 1292)  # Batch of 2
dummy_ids = torch.randint(0, 1000, (2, 512))
dummy_mask = torch.ones(2, 512)

output = model(dummy_spec, dummy_ids, dummy_mask)
print(f"Output shape: {output.shape}")  # Should be [2, 2] (batch, valence/arousal)
print("✅ Model test passed!")
```

**Expected output:**
- Model creates without errors
- Output shape is `[2, 2]`
- No dimension mismatch errors

---

## 🟢 PHASE 3: TRAINING IMPROVEMENTS

### Files to Edit: MODEL_3, MODEL_4, MODEL_5 notebooks
### Cell to Edit: cell-10 (training loop)

---

### FIX 1: Add Gradient Clipping

**Find this section in cell-10:**

```python
        loss.backward()
        optimizer.step()  # ❌ No gradient clipping
```

**Replace with:**

```python
        loss.backward()

        # ✅ Gradient clipping to prevent exploding gradients
        # Especially important when combining CNN + BERT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
```

---

### FIX 2: Track Best Epoch Correctly

**Find this section at the top of the training loop (cell-10):**

```python
# Early Stopping Setup
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_model_state = None
```

**Add this line:**

```python
# Early Stopping Setup
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_model_state = None
best_epoch = 0  # ✅ Track which epoch was best
```

**Then find this section in the validation part:**

```python
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"✓ New best validation loss: {best_val_loss:.4f}")
```

**Replace with:**

```python
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1  # ✅ Record the actual epoch
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"✓ New best validation loss: {best_val_loss:.4f}")
```

**Finally, find this section at the end:**

```python
if patience_counter >= patience:
    print("--- Training Stopped Early ---")
    print(f"Best model restored from epoch {epoch + 1 - patience}")  # ❌ Wrong
else:
    print("--- Training Completed All Epochs ---")
```

**Replace with:**

```python
if patience_counter >= patience:
    print("--- Training Stopped Early ---")
    print(f"Best model restored from epoch {best_epoch}")  # ✅ Correct
else:
    print("--- Training Completed All Epochs ---")
```

---

## 📊 VERIFICATION CHECKLIST

Before retraining, verify each change:

### ✅ Preprocessing (Dissertation_GG.ipynb)
- [ ] `ref=1.0` in power_to_db call
- [ ] Middle segment used for truncation
- [ ] Padding uses `constant_values=-80`
- [ ] Test output shows reasonable dB values

### ✅ SE-Attention (MODEL_3, 4, 5)
- [ ] SEAttention class defined in cell-5
- [ ] SEAttention added to features block BEFORE AdaptiveAvgPool2d
- [ ] Old attention removed from __init__
- [ ] Old attention removed from forward()
- [ ] Model loads without errors

### ✅ Training Improvements (MODEL_3, 4, 5)
- [ ] Gradient clipping added after loss.backward()
- [ ] best_epoch variable initialized
- [ ] best_epoch updated when model improves
- [ ] best_epoch printed at end (not calculated)

---

## 🚀 EXECUTION ORDER

### 1. Fix and Reprocess Data (CRITICAL - DO FIRST)
```
Open: Dissertation_GG.ipynb
Replace: cell-10 (load_song_data function)
Test: cell-11 (verify one song)
Run: cell-12 (reprocess ALL songs) ⚠️ THIS TAKES TIME
```

### 2. Update Model Architecture
```
For each of MODEL_3, MODEL_4, MODEL_5:
  - Replace cell-5 (SEAttention)
  - Replace cell-6 (VGGish_Audio_Model)
  - Test: Instantiate model, verify it loads
```

### 3. Add Training Improvements
```
For each of MODEL_3, MODEL_4, MODEL_5:
  - Edit cell-10 (add gradient clipping)
  - Edit cell-10 (add best_epoch tracking)
```

### 4. Retrain Models
```
Run MODEL_3 training
Run MODEL_4 training
Run MODEL_5 training
Compare results to old models
```

---

## 📈 EXPECTED IMPROVEMENTS

After these fixes, you should see:

1. **Better arousal predictions** (normalization fix allows learning loudness)
2. **Better performance on short songs** (padding fix)
3. **More stable training** (gradient clipping)
4. **Better feature learning** (SE-Attention)
5. **Potential 5-15% improvement in MAE/R²** (combined effect)

---

## 💾 BACKUP REMINDER

Before making changes:
1. Save copies of your current notebooks
2. Save your current best model weights
3. Save current results for comparison

You can revert if needed!

---

## 📝 DISSERTATION NOTES

Document these changes as:

**"Model Refinements Based on Code Review"**
- Corrected spectrogram normalization for consistent loudness representation
- Fixed padding to avoid artificial signals
- Upgraded attention mechanism from feature-level to spatial/channel SE-blocks
- Added gradient clipping for training stability
- Improved truncation strategy to use middle segments

Compare before/after results in a table.

---

## 🎉 AUDIO PROCESSING V2 - FULLY IMPLEMENTED (2025-11-04)

### Notebook: `Audio_Processing_V2_Dissertation_GG.ipynb`

This notebook successfully implements a **global max normalization approach** for dataset-wide consistency.

### ✅ What's Been Implemented

1. **Global Max Power Calculation** (cells 10-15)
   - Computes single maximum power value across ALL spectrograms: `GLOBAL_MAX = 13400.846`
   - Used for dataset-wide normalization consistency
   - Function: `find_global_max_power(dataframe, base_audio_path)`
   - Outlier identified: Song `MT0004073991` in Q2 (helps understand data distribution)

2. **Design Decision: First Segment Processing** ✅
   - Using first 30 seconds of each song (appropriate for ~30s average song length)
   - Truncates longer songs to first 1292 frames (30s at sr=22050, hop=512)
   - No middle-segment extraction needed for this dataset

3. **Correct Padding Implementation** ✅
   - Using `-80 dB` for silence padding (correct!)
   - `constant_values=-80.0` in cell 16, line 40

4. **Normalization Implementation** ✅
   - `ref=GLOBAL_MAX` in cell 16, line 29
   - All spectrograms normalized to same reference (13400.846)
   - Preserves absolute and relative loudness information

5. **Complete Visual Verification** ✅ (NEW - Cell 20)
   - Displays ALL 2216 spectrograms in batches
   - 24 spectrograms per batch (6 columns × 4 rows)
   - 93 total batches
   - **User confirmed: ALL spectrograms verified and look correct!**

### ✅ All Issues Resolved - No Further Action Needed!

**Previous concerns addressed:**
- ✅ Normalization approach decided: `ref=GLOBAL_MAX` (dataset-wide consistency)
- ✅ All 2216 spectrograms reprocessed successfully
- ✅ Visual verification completed for ALL spectrograms
- ✅ Padding correctly implemented with -80 dB (silence)
- ✅ Truncation strategy appropriate for dataset

**Expected dB ranges with GLOBAL_MAX normalization:**
- Most spectrogram values: **-80 to -20 dB** (typical)
- Quieter songs/moments: **-100 to -80 dB**
- Loudest songs: **-10 to 0 dB**
- This is mathematically correct and NOT a bug!
- CNNs can learn effectively from any consistent scale

### 📊 Implementation Summary

| Aspect | Implemented in V2 Notebook |
|--------|---------------------------|
| **Normalization** | ✅ `ref=GLOBAL_MAX` (13400.846) - dataset-wide consistency |
| **Truncation** | ✅ First segment (1292 frames = 30s) |
| **Padding** | ✅ `-80 dB` (silence) |
| **Processing** | ✅ All 2216 songs completed |
| **Verification** | ✅ Visual inspection of ALL spectrograms |
| **Status** | ✅ **PHASE 1 COMPLETE - READY FOR PHASE 2** |

---

**Good luck! Let me know if you hit any issues.**
