# --- 4. MODEL ARCHITECTURE ---

def build_eeg_cnn(input_shape=(18, 256)):
    inputs = layers.Input(shape=input_shape)

    # [IMPORTANT]: Swap axes. Conv1D expects (Time, Channels)
    # Input: (Batch, 18, 256) -> Output: (Batch, 256, 18)
    x = layers.Permute((2, 1))(inputs)

    # Block 1
    x = layers.Conv1D(32, kernel_size=7, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # Block 2
    x = layers.Conv1D(64, kernel_size=5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # Block 3
    x = layers.Conv1D(128, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Classification
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="EEG_CNN_Standard")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    # A. Index Data
    index = get_subject_index(DATASET_ROOT)

    # B. Split Data
    train_files, test_files = get_loso_split(index, TEST_SUBJECT_ID)

    # C. Compute Stats (Strictly on Train files)
    mean, std = compute_global_stats(train_files)
    print(f"Stats Shape - Mean: {mean.shape}, Std: {std.shape}")

    # D. Create Datasets
    train_ds = create_dataset(train_files, mean, std, is_train=True)
    test_ds = create_dataset(test_files, mean, std, is_train=False)

    # E. Train
    model = build_eeg_cnn(input_shape=(N_CHANNELS, EPOCH_LENGTH))
    model.summary()

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        f"model_{TEST_SUBJECT_ID}.h5", save_best_only=True, monitor='val_auc', mode='max'
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc', patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=20,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )