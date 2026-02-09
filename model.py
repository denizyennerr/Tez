# Build Model

def build_cnn_model(input_shape):
    # Expected input_shape: (Time=256, Channels=18)
    model = models.Sequential()

    # Block 1: Temporal Feature Extraction
    model.add(layers.Conv1D(filters=32, kernel_size=64, activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    # Block 2: Spatial & Deep Features
    model.add(layers.Conv1D(filters=64, kernel_size=16, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    # Block 3: Global Features
    model.add(layers.Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'))
    model.add(layers.GlobalAveragePooling1D())

    # Classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    return model

# Initialize Model
# Note: Input shape is (Time, Channels) = (256, 18)
model = build_cnn_model(input_shape=(256, 18))

print("\n🤖 Model Summary:")
model.summary()

# ============================================
# 4. TRAIN MODEL
# ============================================
print("\n🚀 Starting Training...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,  # Adjust as needed
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
)

print("✅ Training Complete.")