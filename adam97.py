# %% Imports and Configuration
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow.keras.metrics import Recall, Precision, AUC


dataset_path = 'master_dataset_2s.npz'
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", "fit", timestamp)

output_dir = os.path.join("saved_outputs", timestamp)
models_dir = os.path.join(output_dir, "models")
histories_dir = os.path.join(output_dir, "histories")

# Create directories explicitly
os.makedirs(models_dir, exist_ok=True)
os.makedirs(histories_dir, exist_ok=True)



# %% Model Definition
def build_seizure_model(input_shape):
    """
    Build CNN-LSTM model with Attention for seizure detection.

    Architecture:
    - Multi-scale CNN for pattern extraction
    - Bidirectional LSTM for temporal learning
    - Attention mechanism to focus on important parts
    - Dense layers for classification
    """

    inputs = layers.Input(shape=input_shape, name='eeg_input')

    # ===== CNN BLOCK: Extract Local Patterns =====
    # Small kernel - catches quick spikes
    conv1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.MaxPooling1D(pool_size=2)(conv1)
    conv1 = layers.Dropout(0.2)(conv1)

    # Medium kernel - catches medium patterns
    conv2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.MaxPooling1D(pool_size=2)(conv2)
    conv2 = layers.Dropout(0.2)(conv2)

    # Larger kernel - catches slower waves
    conv3 = layers.Conv1D(128, kernel_size=7, padding='same', activation='relu')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.MaxPooling1D(pool_size=2)(conv3)
    conv3 = layers.Dropout(0.3)(conv3)

    # ===== LSTM BLOCK: Learn Time Patterns =====
    lstm1 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2)
    )(conv3)

    lstm2 = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.2)
    )(lstm1)

    # ===== ATTENTION BLOCK: Focus on Important Parts =====
    attention = layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.1
    )(lstm2, lstm2)
    attention = layers.LayerNormalization()(attention + lstm2)  # Skip connection

    # ===== POOLING: Combine All Information =====
    avg_pool = layers.GlobalAveragePooling1D()(attention)
    max_pool = layers.GlobalMaxPooling1D()(attention)
    concat = layers.Concatenate()([avg_pool, max_pool])

    # ===== CLASSIFICATION HEAD =====
    dense1 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concat)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.5)(dense1)

    dense2 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dense1)
    dense2 = layers.Dropout(0.3)(dense2)

    # Output: probability of seizure
    outputs = layers.Dense(1, activation='sigmoid', name='seizure_output')(dense2)

    model = Model(inputs=inputs, outputs=outputs, name='SeizureDetector')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    print("\n⚙️ Model Compiled:")
    print("   • Optimizer: Adam (lr=0.001)")
    print("   • Loss: Binary Crossentropy")
    print("   • Metrics: Accuracy, AUC, Precision, Recall")

    return model


# %% Data Loading
data = np.load(dataset_path)
X, y, groups = data['X'], data['y'], data['s']

logo = LeaveOneGroupOut()
input_shape = (X.shape[1], X.shape[2])

print(f"Total Data Shape: {X.shape} | Subjects: {np.unique(groups)}")
print(f"Outputs will be saved to: {output_dir}")

# %% LOSO Training Loop
for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    current_test_subject = groups[test_idx][0]
    print(f"\n🚀 Training holding out Subject: {current_test_subject}")

    # Subject-level Validation Split
    unique_train_subjects = np.unique(groups_train)
    val_subject = unique_train_subjects[0]
    val_mask = (groups_train == val_subject)
    train_mask = ~val_mask

    X_train, X_val = X_train_full[train_mask], X_train_full[val_mask]
    y_train, y_val = y_train_full[train_mask], y_train_full[val_mask]

    model = build_seizure_model(input_shape)
    fold_log_dir = os.path.join(log_dir, f"subject_{current_test_subject}")

    # --- NEW: ModelCheckpoint added to callbacks ---
    model_save_path = os.path.join(models_dir, f"best_model_subject_{current_test_subject}.keras")

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        TensorBoard(log_dir=fold_log_dir, histogram_freq=1),
        ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # --- NEW: Save training history to CSV for later plotting ---
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(histories_dir, f"history_subject_{current_test_subject}.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"✅ Saved history for subject {current_test_subject}")

print("\n🎉 All training complete. Models and histories saved successfully.")