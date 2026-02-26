# %% Imports and Configuration
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
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
def build_seizure_model(input_shape, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # L2 Regularizer
    reg = regularizers.l2(0.001)

    model = models.Sequential([
        Input(shape=input_shape),
        # Block 1: Capture low-level features
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.SpatialDropout1D(0.35),

        # Block 2: Mid-level features
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.SpatialDropout1D(0.35),

        # Block 3: High-level features
        # layers.Dropout(0.5),  # Aggressive dropout before classifier
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Recall(name='recall'), Precision(name='precision')]
    )
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
        epochs=250,
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