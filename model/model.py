from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
import os
import numpy as np
from sklearn.model_selection import train_test_split

def create_model(num_classes, seq_length=80, feature_dim=1663, 
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
    sign_classes = os.listdir('/mnt/hdd/Downloads/keypoint_data_selected_augmented')
    
    x, y = [], []

    for label, gesture in enumerate(sign_classes):
        gesture_dir = os.path.join('/mnt/hdd/Downloads/keypoint_data_selected_augmented', gesture)
        files = [os.path.join(gesture_dir, f) for f in os.listdir(gesture_dir) if f.endswith(".npy")]
        
        for f in files:
            x.append(f)
            y.append(label)

# Split at the video level — ensures no data leakage
    X_train_files, X_test_files, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    for f in X_train_files:
        data = np.load(f)

# Load the actual numpy arrays
    X_train = np.array([np.load(f) for f in X_train_files])
    X_test = np.array([np.load(f) for f in X_test_files])

    y_train = np.array(y_train)
    y_test = np.array(y_test)


    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)


    model  = create_model(num_classes=len(sign_classes))


    model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )

    history = model.fit(
            X_train,
            y_train,
            )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save("sign_lstm_transformer_model.keras")

    

                  

