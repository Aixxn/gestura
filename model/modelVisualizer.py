from tensorflow.keras import layers, models, utils

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

# Create and visualize the model
model = create_model(num_classes=10)
utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file="lstm_transformer_model.png")

