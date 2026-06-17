# Gestura ML Pipeline

## Overview

The ML pipeline converts ASL video → MediaPipe keypoints → normalized 35-frame windows → classification via a BiLSTM+Transformer model. The trained model is deployed to the Translation Service for real-time inference.

---

## Feature Vector (258 dimensions)

Each frame produces a 258-dimensional feature vector from MediaPipe Holistic:

| Component | Landmarks | Dimensions | Index Range |
|-----------|-----------|------------|-------------|
| Left Hand | 21 | 21 × 3 (x, y, z) = 63 | `[0:63]` |
| Right Hand | 21 | 21 × 3 (x, y, z) = 63 | `[63:126]` |
| Pose Coords | 33 | 33 × 3 (x, y, z) = 99 | `[126:225]` |
| Pose Visibility | 33 | 33 × 1 = 33 | `[225:258]` |

**Face landmarks are excluded** — the model does not use them.

All coordinates are **nose-normalized**: each frame subtracts the nose landmark (pose index 0) xyz from all hand and pose coordinates. This makes the features translation-invariant.

---

## Data Extraction (`extract_data.py`)

### Input Layout
```
videos/
├── HELLO/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── THANK_YOU/
│   ├── video1.mp4
│   └── ...
└── ...
```

### Output Layout
```
keypoint_data/
├── HELLO/
│   ├── 0.npy   # shape (35, 258)
│   ├── 1.npy
│   └── ...
├── THANK_YOU/
│   ├── 0.npy
│   └── ...
└── ...
```

### Extraction Process
1. Read video file using OpenCV
2. For each frame, run MediaPipe Holistic (using `mp.solutions.holistic` API — stable version, avoids GPU driver crashes on Intel Mesa)
3. Extract 258-dim feature vector
4. Apply nose-normalization
5. Collect consecutive frames as signs (using background collection for non-sign frames)
6. Resample to uniform 35-frame windows
7. Save each window as `.npy` file named by index

### Background Collection
The extractor also collects "BACKGROUND" samples — periods where no ASL sign is being performed (hands resting, transition between signs). These are sampled and labeled with a dedicated `BACKGROUND` class so the model learns to distinguish silence from actual signs.

---

## Data Augmentation (`DataAugmentor.py`)

Applied during training to improve generalization. Only coordinate columns `[0:225]` are transformed; pose visibility `[225:258]` is preserved.

| Technique | Range | Description |
|-----------|-------|-------------|
| Jitter | σ = 0.004 | Gaussian noise added to landmark positions |
| Scaling | 0.93–1.07 | Uniform scaling of all coordinates |
| Rotation | ±7° | Small rotations in 3D space |
| Translation | ±0.05 | Small shifts in xyz |
| Mirror | 50% prob | Horizontal flip (swaps left/right hands) |

The augmentor splits the (35, 258) tensor into coordinates (35, 225) and visibility (35, 33), reshapes coordinates to (35, 75, 3) for spatial transforms, then flattens back.

---

## Model Architecture (`model.py`)

### BiLSTM + Transformer

```
Input: (35, 258)
    │
    ▼
Bidirectional LSTM (256 units, return_sequences=True)
    │  ┌────────────────────────────┐
    │  │ Captures temporal patterns  │
    │  │ in both forward/backward    │
    │  │ directions                  │
    │  └────────────────────────────┘
    │
    ▼
Dense Projection (256 → d_model=256)
    │
    ▼
Layer Normalization
    │
    ▼
Multi-Head Self-Attention (num_heads=4)
    │  ┌────────────────────────────┐
    │  │ Attends over all 35        │
    │  │ timesteps to find which    │
    │  │ frames are most important  │
    │  └────────────────────────────┘
    │
    ▼
Add & Layer Normalization (residual connection)
    │
    ▼
Feed-Forward Network (256 → 512 → 256, ReLU)
    │
    ▼
Add & Layer Normalization (residual connection)
    │
    ▼
Global Average Pooling 1D
    │
    ▼
Dense (256, ReLU)
    │
    ▼
Dropout (0.4)
    │
    ▼
Dense (num_classes, Softmax)
    │
    ▼
Output: class probabilities
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 35 frames |
| Feature Dimension | 258 |
| LSTM Units | 256 (bidirectional) |
| Transformer d_model | 256 |
| Attention Heads | 4 |
| Feed-Forward Dim | 512 |
| Dropout | 0.4 |
| Learning Rate | 1e-4 (Adam) |
| Batch Size | 32 |
| Epochs | 150 (with early stopping) |
| Early Stopping | patience=20, monitor=val_accuracy |
| LR Reduction | factor=0.5, patience=7, min=1e-6 |
| Weighted Loss | Balanced class weights (clipped to 3× max) |

### Training Flow
1. Load `.npy` files from `keypoint_data_augmented/`
2. Split 80/20 train/test (stratified)
3. Compute balanced class weights
4. Train with callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
5. Evaluate: overall accuracy + per-class accuracy
6. Sanity check: all-zeros input → should predict BACKGROUND
7. Save: `best_model.keras`, `sign_lstm_transformer_model.keras`, `sign_classes.npy`

---

## Inference Pipeline (Translation Service)

### At Runtime
1. **MediaPipe Holistic** extracts 258-dim keypoints from each camera frame
2. **Bounded Persistence** fills in last-known hand positions for up to 5 lost frames, then decays to zeros
3. **Nose-normalization** subtracts nose xyz for translation invariance
4. **Exponential Moving Average** (alpha=0.4) smooths micro-jitter
5. **Motion Detector** computes frame-to-frame motion magnitude, detects sign boundaries via hysteresis
6. On sign boundary: the accumulated keypoint sequence is **normalized to exactly 35 frames** (resampled if longer, padded with last frame if shorter)
7. **model.predict()** returns class probabilities; argmax → word from `sign_classes.npy`
8. "BACKGROUND" predictions are filtered out (not real signs)
9. Words accumulate in session state
10. On session end: **FLAN-T5** converts ASL gloss → grammatical English

### Normalization (`normalize.py`)
- If `len(frames) > 35`: Uniformly downsample to 35 frames
- If `len(frames) < 35`: Pad with copies of the last frame
- If `len(frames) == 35`: Pass through unchanged

### File Locations
- Model: `translationService/best_model.keras` (also `model/best_model.keras`)
- Classes: `translationService/sign_classes.npy` (also `model/sign_classes.npy`)
- MediaPipe model: `translationService/holistic_landmarker.task` (auto-downloaded if missing)

---

## Integration Test (`scripts/test_pipeline.py`)

A standalone script that runs the full pipeline with webcam input:

```bash
source translationService/.venv/bin/activate
python scripts/test_pipeline.py
```

**Controls:**
- `Q` / `ESC` — Stop session, run grammar correction, print result
- `C` — Clear accumulated words and restart
- `D` — Toggle debug overlay (motion detector internals)

**Output includes:**
- Real-time detected words with confidence
- Performance summary (signs/minute, avg duration, avg confidence)
- Segmentation metrics (boundary count, skipped backgrounds)
- Final ASL gloss → English translation

---

## Model Files

| File | Size | Location | Description |
|------|------|----------|-------------|
| `best_model.keras` | ~21 MB | `model/`, `translationService/` | Best checkpoint (by val_accuracy) |
| `model.keras` | ~41 MB | `model/`, `translationService/` | Older model variant |
| `sign_lstm_transformer_model.keras` | ~21 MB | `model/` | Post-training save |
| `sign_classes.npy` | ~1.3 KB | `model/`, `translationService/` | Class name mapping |
| `holistic_landmarker.task` | ~13 MB | `translationService/` | MediaPipe model (auto-downloaded) |

---

## Data Directory Structure

```
model/
├── keypoint_data/              # Raw extracted keypoints
│   ├── HELLO/
│   │   ├── 0.npy
│   │   └── ...
│   ├── THANK_YOU/
│   └── BACKGROUND/
├── keypoint_data_augmented/   # Original + augmented copies
│   └── (same structure)
├── keypoint_data_normalized/  # Post-normalization copies
│   └── (same structure)
├── best_model.keras
├── sign_classes.npy
└── ...
```

**Note:** The `model/` directory generates the trained artifacts. The `translationService/` directory consumes them (copies of `best_model.keras` and `sign_classes.npy`).
