# Implementation Guide: Critical Fixes and SE-Attention Upgrade

**Date:** 2025-10-30
**Purpose:** Complete code fixes for preprocessing bugs and SE-Attention implementation

---

## 📋 TODO LIST

### 🔴 PHASE 1: CRITICAL PREPROCESSING FIXES (DO FIRST!)

- [ ] **Task 1.1:** Fix `load_song_data()` function in `Dissertation_GG.ipynb` (cell-10)
- [ ] **Task 1.2:** Verify the fix by testing on one song
- [ ] **Task 1.3:** Reprocess ALL spectrograms with fixed code (cell-12)
- [ ] **Task 1.4:** Verify reprocessed data looks correct

**Estimated time:** 2-3 hours (depending on dataset size)

### 🟡 PHASE 2: MODEL ARCHITECTURE UPGRADES (DO SECOND)

- [ ] **Task 2.1:** Add `SEAttention` class to MODEL_3, 4, 5 (cell-5)
- [ ] **Task 2.2:** Update `VGGish_Audio_Model` in MODEL_3, 4, 5 (cell-6)
- [ ] **Task 2.3:** Verify model architecture loads correctly

**Estimated time:** 30 minutes

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

## 🟡 PHASE 2: SE-ATTENTION IMPLEMENTATION

### Files to Edit: MODEL_3, MODEL_4, MODEL_5 notebooks

---

### STEP 1: Replace cell-5 (AttentionModule) with SEAttention

**File:** `MODEL_3_Dissertation_model_training_basic_VGGish.ipynb`
**Cell:** cell-5

**BEFORE (old AttentionModule):**
```python
class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted_features = x * attention_weights
        return weighted_features
```

**AFTER (new SEAttention) - REPLACE ENTIRE CELL:**

```python
class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention Block

    This module performs channel-wise attention on CNN feature maps.
    It learns to re-weight channels based on global information,
    allowing the network to focus on the most important features.

    Reference: "Squeeze-and-Excitation Networks" (Hu et al., 2018)

    Args:
        channel_dim: Number of channels in the input feature map
        reduction_ratio: Reduction factor for the bottleneck (default: 16)

    Input shape: [batch_size, channel_dim, height, width]
    Output shape: [batch_size, channel_dim, height, width] (same as input)
    """
    def __init__(self, channel_dim, reduction_ratio=16):
        super(SEAttention, self).__init__()

        # Squeeze: Global Average Pooling (spatial dimensions -> 1x1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation: Two-layer MLP with bottleneck
        self.excitation = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_dim // reduction_ratio, channel_dim, bias=False),
            nn.Sigmoid()  # Output values between 0 and 1 (channel weights)
        )

    def forward(self, x):
        """
        Forward pass of SE block

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Attention-weighted feature map [B, C, H, W]
        """
        # Get dimensions
        batch_size, channels, _, _ = x.shape

        # Squeeze: [B, C, H, W] -> [B, C, 1, 1]
        # This captures global information from each channel
        squeezed = self.squeeze(x)

        # Flatten for MLP: [B, C, 1, 1] -> [B, C]
        squeezed = squeezed.view(batch_size, channels)

        # Excitation: [B, C] -> [B, C]
        # Learns the importance weight for each channel
        channel_weights = self.excitation(squeezed)

        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1]
        channel_weights = channel_weights.view(batch_size, channels, 1, 1)

        # Apply channel weights: [B, C, H, W] * [B, C, 1, 1] -> [B, C, H, W]
        # Each channel is scaled by its learned weight
        return x * channel_weights.expand_as(x)
```

**Repeat this for MODEL_4 and MODEL_5 (same code)**

---

### STEP 2: Update VGGish_Audio_Model (cell-6)

**File:** `MODEL_3_Dissertation_model_training_basic_VGGish.ipynb`
**Cell:** cell-6

**BEFORE:**
```python
class VGGish_Audio_Model(nn.Module):
    def __init__(self):
        super(VGGish_Audio_Model, self).__init__()

        self.features = nn.Sequential(
            # ... conv blocks ...
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # ❌ Attention should be BEFORE this
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64)
        )

        self.attention = AttentionModule(64)  # ❌ Too late!

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.attention(x)  # ❌ Applied after everything
        return x
```

**AFTER - REPLACE ENTIRE CELL:**

```python
class VGGish_Audio_Model(nn.Module):
    """
    VGG-style CNN for audio feature extraction with SE-Attention

    V2.0 Changes:
    - Moved attention from final features to feature maps (before pooling)
    - Now uses SEAttention instead of simple feature-level attention
    - Attention can now learn spatial importance in spectrograms

    Architecture:
    - 4 convolutional blocks with increasing channels (64 -> 128 -> 256 -> 512)
    - Batch normalization after each conv
    - Max pooling after blocks 1-3
    - SE-Attention after block 4 (before adaptive pooling)
    - Adaptive average pooling to [B, 512, 1, 1]
    - MLP classifier: 512 -> 256 -> 64

    Input: [batch_size, 1, 128, 1292] (mel spectrogram)
    Output: [batch_size, 64] (audio feature vector)
    """

    def __init__(self):
        super(VGGish_Audio_Model, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 64 channels
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # ✅ SE-ATTENTION APPLIED HERE (before pooling)
            # Operates on [B, 512, H, W] feature maps
            # Learns which channels (feature types) are most important
            SEAttention(channel_dim=512, reduction_ratio=16),

            # Now pool the attention-weighted features
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 512, H, W] -> [B, 512, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64)  # Output: 64-dim audio features
        )

        # ✅ REMOVED: self.attention = AttentionModule(64)
        # Attention is now part of self.features

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [batch_size, 1, 128, 1292] spectrogram

        Returns:
            [batch_size, 64] audio feature vector
        """
        # CNN + SE-Attention: [B, 1, 128, 1292] -> [B, 512, 1, 1]
        x = self.features(x)

        # Flatten: [B, 512, 1, 1] -> [B, 512]
        x = x.view(x.size(0), -1)

        # Classifier: [B, 512] -> [B, 64]
        x = self.classifier(x)

        # ✅ REMOVED: x = self.attention(x)

        return x
```

**Repeat this for MODEL_4 and MODEL_5**

**NOTE FOR MODEL_4:** MODEL_4 uses 2 conv layers per block. Keep that, just add SE-Attention:

```python
# For MODEL_4, the features block should be:
self.features = nn.Sequential(
    # Block 1 - 2 convolutions
    nn.Conv2d(1, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),  # ← MODEL_4 has this extra conv
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Block 2 - 2 convolutions
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=3, padding=1),  # ← MODEL_4 has this extra conv
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Block 3 - 2 convolutions
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),  # ← MODEL_4 has this extra conv
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Block 4 - 2 convolutions
    nn.Conv2d(256, 512, kernel_size=3, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),  # ← MODEL_4 has this extra conv
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),

    # ✅ SE-Attention (same for MODEL_4)
    SEAttention(channel_dim=512, reduction_ratio=16),

    nn.AdaptiveAvgPool2d((1, 1))
)
```

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

**Good luck! Let me know if you hit any issues.**
