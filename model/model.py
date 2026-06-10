from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
import os
import numpy as np
from sklearn.model_selection import train_test_split
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
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
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
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss    : {test_loss:.4f}")

    model.save("sign_lstm_transformer_model.keras")
    np.save("sign_classes.npy", np.array(sign_classes))
    print("\nModel and class map saved.")
    print("Load classes later with: np.load('sign_classes.npy', allow_pickle=True)")