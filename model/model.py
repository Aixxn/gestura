from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import Sequence

# FIX: Corrected comment — face landmarks are NOT in the feature vector.
# Feature layout: lh(63) + rh(63) + pose_coords(99) + pose_vis(33) = 258
SEQ_LENGTH  = 35
FEATURE_DIM = 258

class NumpyArrayGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True):
        self.file_paths = list(file_paths)
        self.labels     = list(labels)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index   = (index + 1) * self.batch_size

        batch_files  = self.file_paths[start_index:end_index]
        batch_labels = self.labels[start_index:end_index]

        X = np.array([np.load(f).astype(np.float32) for f in batch_files])
        y = np.array(batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.file_paths))
            np.random.shuffle(indices)
            self.file_paths = [self.file_paths[i] for i in indices]
            self.labels     = [self.labels[i]     for i in indices]

def create_model(num_classes, seq_length=SEQ_LENGTH, feature_dim=FEATURE_DIM,
                 lstm_units=256, d_model=256, num_heads=4,
                 ff_dim=512, dropout=0.4):
    inputs = layers.Input(shape=(seq_length, feature_dim))

    # Bidirectional LSTM — captures temporal direction both ways
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)

    # Project to transformer dimension
    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Transformer block
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = models.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(d_model),
    ])
    ffn_output = ffn(x)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global pooling + classifier
    x       = layers.GlobalAveragePooling1D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    DATA_ROOT  = os.path.join(os.path.dirname(__file__), "keypoint_data_augmented")
    BATCH_SIZE = 32
    EPOCHS     = 150

    # Collect file paths and labels
    sign_classes = sorted(os.listdir(DATA_ROOT))  # sorted for reproducibility
    x, y = [], []

    for label, gesture in enumerate(sign_classes):
        gesture_dir = os.path.join(DATA_ROOT, gesture)
        if not os.path.isdir(gesture_dir):
            continue
        for fname in os.listdir(gesture_dir):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(gesture_dir, fname)
            try:
                arr = np.load(fpath)
                if arr.shape != (SEQ_LENGTH, FEATURE_DIM):
                    print(f"  ⚠️  Skipping {fpath}: shape {arr.shape}")
                    continue
            except Exception as e:
                print(f"  ⚠️  Could not load {fpath}: {e}")
                continue
            x.append(fpath)
            y.append(label)

    print(f"\nTotal samples: {len(x)}  |  Classes: {len(sign_classes)}")
    print("Classes:", sign_classes)

    # FIX: Guard against empty dataset — if shape mismatches caused all files to
    # be skipped, training will silently produce garbage results without this check.
    if len(x) == 0:
        raise RuntimeError(
            "No valid .npy files found. "
            f"Expected shape ({SEQ_LENGTH}, {FEATURE_DIM}). "
            "Re-run extraction.py to regenerate keypoints."
        )

    X_train_files, X_test_files, y_train_labels, y_test_labels = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train: ({len(X_train_files)}, {SEQ_LENGTH}, {FEATURE_DIM}) ({len(y_train_labels)},)")
    print(f"Test:  ({len(X_test_files)}, {SEQ_LENGTH}, {FEATURE_DIM}) ({len(y_test_labels)},)")

    train_generator = NumpyArrayGenerator(
        file_paths=X_train_files,
        labels=y_train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_generator = NumpyArrayGenerator(
        file_paths=X_test_files,
        labels=y_test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = create_model(num_classes=len(sign_classes))
    model.summary()

    # FIX: Lowered learning rate from 1e-3 → 1e-4.
    # BiLSTM+Transformer stacks are unstable at 1e-3; the model diverges
    # in the first few epochs and never recovers meaningful gradients.
    # Class weights for balanced training (handle remaining imbalance)
    unique_labels = sorted(set(y_train_labels))
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.array(unique_labels), y=np.array(y_train_labels)
    )
    class_weight_dict = dict(zip(unique_labels, class_weights_arr))
    max_w = max(class_weight_dict.values())
    # Clip extreme weights to avoid over-emphasising tiny classes
    class_weight_dict = {k: min(v, max_w * 3) for k, v in class_weight_dict.items()}

    if len(sign_classes) > 30:
        bg_idx = sign_classes.index("BACKGROUND")
        print(f"\n  BACKGROUND is class index {bg_idx}")
        print(f"  BACKGROUND weight: {class_weight_dict.get(bg_idx, 1):.3f}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    checkpoint     = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
    reduce_lr      = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\n{'='*60}")
    print(f"  Overall test accuracy: {test_acc:.4f}")
    print(f"  Overall test loss    : {test_loss:.4f}")
    print(f"{'='*60}")

    # ---- Per-class accuracy ----
    print(f"\n{'='*60}")
    print("  Per-class accuracy on test set:")
    print(f"{'='*60}")
    all_preds = []
    all_true  = []
    for batch_idx in range(len(test_generator)):
        Xb, yb = test_generator[batch_idx]
        pb = model.predict(Xb, verbose=0)
        preds_b = np.argmax(pb, axis=1)
        all_preds.extend(preds_b.tolist())
        all_true.extend(yb.tolist())
    all_preds = np.array(all_preds)
    all_true   = np.array(all_true)
    for cls_idx, cls_name in enumerate(sign_classes):
        mask = all_true == cls_idx
        if mask.sum() > 0:
            acc = (all_preds[mask] == cls_idx).mean()
            print(f"    {cls_name:12s} ({cls_idx:2d}): {acc:.2%}  ({int(mask.sum())} samples)")
        else:
            print(f"    {cls_name:12s} ({cls_idx:2d}): N/A (no test samples)")

    # ---- Zero-input sanity check ----
    print(f"\n{'='*60}")
    print("  Zero-input sanity check (all-zeros → should be BACKGROUND):")
    zero_inp = np.zeros((1, SEQ_LENGTH, FEATURE_DIM), dtype=np.float32)
    zero_probs = model.predict(zero_inp, verbose=0)[0]
    zero_idx = int(np.argmax(zero_probs))
    print(f"    Predicted: {sign_classes[zero_idx]} @ {zero_probs[zero_idx]:.2%}")
    if len(sign_classes) > 30 and zero_idx == sign_classes.index("BACKGROUND"):
        print("    ✅ Zero input correctly classified as BACKGROUND")
    elif len(sign_classes) > 30:
        bg_idx = sign_classes.index("BACKGROUND")
        print(f"    ⚠️  Expected BACKGROUND (idx {bg_idx}), got {sign_classes[zero_idx]} (idx {zero_idx})")
        print(f"    BACKGROUND probability: {zero_probs[bg_idx]:.4%}")
    print(f"{'='*60}")

    # ---- Save ----
    model.save("sign_lstm_transformer_model.keras")
    np.save("sign_classes.npy", np.array(sign_classes))
    print("\n✅ Model and class map saved.")
    print("   Files: sign_lstm_transformer_model.keras, best_model.keras, sign_classes.npy")
    print("   Load classes later with: np.load('sign_classes.npy', allow_pickle=True)")