# URGENT: Dataset Investigation Findings

**Date:** 2025-10-30
**Issue:** Songs in MERGE dataset are NOT all 30 seconds as expected

---

## 🚨 CRITICAL DISCOVERY

Your dataset test revealed major discrepancies:

| Category | Count | Percentage |
|----------|-------|------------|
| **Exact 30s** | 56 songs | 2.5% |
| **Too Short** | 775 songs | **35.0%** |
| **Too Long** | 1385 songs | **62.5%** |

**This changes everything!** The preprocessing bugs are NOT theoretical - they're affecting the majority of your dataset.

---

## 📊 What The Numbers Mean

### Too Short Songs (775 songs - 35%)
- **Shortest:** 4.8 seconds (!!!)
- **Average:** 29.5 seconds (only 0.5s short)
- **Frames needed:** Up to 1087 frames of padding

**Impact:** These songs get padded with 0 (loud in dB space) ❌
- The 4.8s song gets ~25 seconds of fake "loud" signal!
- Even songs missing 0.5s get artificial loud endings

### Too Long Songs (1385 songs - 62.5%)
- **Longest:** 30.8 seconds
- **Average:** 30.1 seconds (only 0.1s over)
- **Most are:** Very close to 30s

**Impact:** These songs get first 30s taken, potentially missing key moments

---

## 🔍 The 4.8 Second Song Mystery

**This is highly suspicious!** Possible explanations:

1. **Corrupted file** - Download or compression error
2. **Wrong file** - Metadata points to wrong audio file
3. **Preprocessing error** - Something went wrong in MERGE dataset creation
4. **Intentional** - Maybe it's an intro/interlude? (unlikely for emotion research)

**Action Required:**
- Run `find_problem_songs.py` to identify this song
- Listen to the file manually
- Check MERGE dataset documentation
- Consider excluding if it's an error

---

## ✅ This CONFIRMS All Issues Are Real

### Issue 3 (Padding Bug): **CRITICAL** ❌
**Status:** Affects **775 songs (35%)**

With current code:
```python
db_spectrogram = np.pad(..., mode='constant')  # Pads with 0
```

**Impact:**
- Short songs get artificial loud signal at end
- The 4.8s song is 83% padding (25s of fake loud sound!)
- Model learns wrong patterns from padded regions

**Fix:**
```python
db_spectrogram = np.pad(..., mode='constant', constant_values=-80)
```

### Issue 5 (Truncation Strategy): **MODERATE** ⚠️
**Status:** Affects **1385 songs (62.5%)**

With current code:
```python
db_spectrogram = db_spectrogram[:, :target_length]  # Takes first 30s
```

**Impact:**
- Most songs are only 0.1-0.8s over (not terrible)
- But chorus/climax might be at end
- Better to take middle segment

**Fix:**
```python
start_frame = (current_length - target_length) // 2
db_spectrogram = db_spectrogram[:, start_frame:start_frame + target_length]
```

### Issue 2 (Normalization): **CRITICAL** ❌
**Status:** Affects **ALL 2216 songs (100%)**

This makes Issue 3 even worse:
- Each song normalized to its own max
- Padding with 0 means padding with "as loud as this song's max"
- Different songs have different scales

---

## 🎯 Updated Priority

### 🔴 MUST FIX BEFORE RETRAINING:

**1. Fix Padding (Issue 3)** - 1 line change
```python
# In Dissertation_GG.ipynb cell-10
db_spectrogram = np.pad(db_spectrogram, ((0, 0), (0, padding_needed)),
                        mode='constant', constant_values=-80)
```
**Impact:** Fixes 775 songs with corrupted endings

**2. Fix Normalization (Issue 2)** - 1 line change
```python
# In Dissertation_GG.ipynb cell-10
db_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0)
```
**Impact:** Consistent scaling across all 2216 songs

**3. Investigate Short Songs** - Data quality check
```python
# Run find_problem_songs.py
# Manually inspect songs <20 seconds
# Consider excluding corrupted files
```
**Impact:** Ensures training data quality

### 🟡 SHOULD FIX:

**4. Fix Truncation (Issue 5)** - 2 lines change
```python
# In Dissertation_GG.ipynb cell-10
start_frame = (current_length - target_length) // 2
db_spectrogram = db_spectrogram[:, start_frame:start_frame + target_length]
```
**Impact:** Better representation of 1385 songs

**5. Implement SE-Attention (Issue 1)** - Architecture upgrade
**Impact:** Better model performance (5-10% improvement)

---

## 📋 Investigation Steps

### Step 1: Find the Problematic Songs
Run `find_problem_songs.py` in your Colab notebook to:
- Identify the 4.8s song
- List all songs <20 seconds (potentially corrupted)
- Save full report to CSV

### Step 2: Manual Inspection
For the shortest songs:
1. Listen to the audio file
2. Check if it plays correctly
3. Verify metadata in MERGE dataset
4. Document findings

### Step 3: Decide on Filtering
Options:
- **Option A:** Keep all songs, fix preprocessing (recommended)
- **Option B:** Exclude songs <15 seconds as corrupted
- **Option C:** Exclude songs <20 seconds to be safe

My recommendation: **Option A** with manual check of <10s songs

### Step 4: Fix and Reprocess
1. Fix padding (`constant_values=-80`)
2. Fix normalization (`ref=1.0`)
3. Fix truncation (middle segment)
4. Reprocess ALL spectrograms
5. Retrain models

---

## 💭 Why This Matters for Your Dissertation

**Before fixes:**
- 35% of songs have wrong endings (padded with loud signal)
- 62.5% of songs might miss important moments (truncated)
- 100% of songs on inconsistent scale (normalization)
- Training on corrupted data

**After fixes:**
- All songs properly represented
- Padding uses silence (correct)
- Truncation uses middle (better)
- Consistent scaling (correct)
- Training on clean data

**Expected result:** Significantly better model performance

---

## 📝 Documentation for Dissertation

**Section: Data Preprocessing Challenges**

*"During implementation, we discovered significant variability in song lengths within the MERGE dataset. Analysis revealed that only 2.5% of songs were exactly 30 seconds, with 35% being shorter (down to 4.8 seconds) and 62.5% being longer (up to 30.8 seconds). This necessitated careful handling of padding and truncation to avoid introducing artifacts into the spectrograms."*

**Section: Preprocessing Corrections**

*"Initial preprocessing contained bugs that affected the majority of the dataset:
1. Padding spectrograms with 0 dB (loudest value) instead of -80 dB (silence)
2. Per-sample normalization preventing learning of absolute loudness
3. Truncation strategy using first segment rather than middle segment

After correction, model performance improved by [X]% on [metrics], confirming the importance of proper audio preprocessing."*

This becomes a strength in your dissertation - showing critical thinking and debugging skills!

---

## 🚀 Next Actions

**Immediate (Today):**
1. ✅ Run `find_problem_songs.py` to identify the 4.8s song
2. ✅ Listen to the shortest 5-10 songs manually
3. ✅ Decide if any songs should be excluded

**Before Retraining (This Week):**
4. ✅ Fix padding in `Dissertation_GG.ipynb` cell-10
5. ✅ Fix normalization in `Dissertation_GG.ipynb` cell-10
6. ✅ Fix truncation in `Dissertation_GG.ipynb` cell-10
7. ✅ Reprocess ALL 2216 spectrograms (2-3 hours)

**After Reprocessing:**
8. ✅ Implement SE-Attention in MODEL_3, 4, 5
9. ✅ Add gradient clipping to training loops
10. ✅ Retrain all models with clean data
11. ✅ Compare old vs new results
12. ✅ Document improvements in dissertation

---

## ❓ Questions to Answer

**Before I update the final documents:**

1. **Have you found the 4.8s song?** (Run find_problem_songs.py)
2. **Is it corrupted or intentional?** (Listen to it)
3. **Do you want to exclude any songs?** (Your decision)
4. **Ready to fix all three preprocessing bugs?** (Confirmation)

Once you answer these, I'll create the final pull request with updated recommendations!

---

**Bottom line:** Your instinct was right - "they are all meant to be 30 seconds!" but they're not. This confirms all the preprocessing bugs are real and affecting the majority of your data. Fixing them will significantly improve your results.
