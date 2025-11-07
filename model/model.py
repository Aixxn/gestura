from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

class NumpyArrayGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        # Get batch start and end indices
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        
        # Get the file paths for the current batch
        batch_files = self.file_paths[start_index:end_index]
        batch_labels = self.labels[start_index:end_index]

        # Load data from disk for the current batch
        # We ensure it's float32 and then convert to a numpy array
        X = np.array([np.load(f).astype(np.float32) for f in batch_files])
        y = np.array(batch_labels)
        
        return X, y

    def on_epoch_end(self):
        # Optional: Shuffle data indices at the end of each epoch
        if self.shuffle:
            indices = np.arange(len(self.file_paths))
            np.random.shuffle(indices)
            self.file_paths = [self.file_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

def create_model(num_classes, seq_length=80, feature_dim=1662, 
                                  lstm_units=256, d_model=256, num_heads=4,
                                  ff_dim=512, dropout=0.3):
    inputs = layers.Input(shape=(seq_length, feature_dim))

    # LSTM part
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)

    # Project to transformer dimension
    x = layers.Dense(d_model)(x)

    # Transformer block
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = models.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(d_model),
    ])
    ffn_output = ffn(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global pooling + classifier
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    sign_classes = os.listdir('/mnt/hdd/Downloads/data')
    x, y = [], []
    for label, gesture in enumerate(sign_classes):
        gesture_dir = os.path.join('/mnt/hdd/Downloads/data', gesture)
        files = [os.path.join(gesture_dir, f) for f in os.listdir(gesture_dir) if f.endswith(".npy")]
        for f in files:
            x.append(f)
            y.append(label)
    X_train_files, X_test_files, y_train_labels, y_test_labels = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )


    # 1. Instantiate the generators
    BATCH_SIZE = 32 # Keep the batch size from your original fit call
    train_generator = NumpyArrayGenerator(
        file_paths=X_train_files,
        labels=y_train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # For the test/validation set, you typically don't shuffle
    test_generator = NumpyArrayGenerator(
        file_paths=X_test_files,
        labels=y_test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Note: We don't need to load the full X_train or X_test arrays anymore.
    # The printed shapes will now reflect the *number of files* and *labels*
    print(f"Train: ({len(X_train_files)}, 80, 1663) ({len(y_train_labels)},)")
    print(f"Test: ({len(X_test_files)}, 80, 1663) ({len(y_test_labels)},)")

    # 2. Instantiate and compile the model
    model  = create_model(num_classes=len(sign_classes))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')

    # 3. Use .fit() with the generators
    history = model.fit(
        # Pass the generator instead of the full NumPy arrays
        train_generator,
        validation_data=test_generator,
        epochs=150,
        callbacks=[early_stopping, checkpoint],
        verbose=1
        # batch_size is now controlled by the generator
    )
    
    # Use model.evaluate with the generator
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")

    # Final model save
    model.save("sign_lstm_transformer_model.keras")
