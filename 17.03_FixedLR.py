# %% Imports and Configuration
import os
import datetime
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow.keras.backend as K  # Added for memory management

# Check for GPUs and set up MirroredStrategy for Multi-GPU training
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"✅ GPUs Detected: {len(physical_devices)}")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    # This automatically detects and uses all available GPUs
    strategy = tf.distribute.MirroredStrategy()
    print(f"🚀 Training distributed across {strategy.num_replicas_in_sync} GPUs")
else:
    print("⚠️ WARNING: No GPU detected. Running on CPU.")
    strategy = tf.distribute.get_strategy()  # Default strategy (CPU)

# ==========================================
# %% Constants & Dataset Paths
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 100
DECISION_THRESHOLD = 0.3

dataset_paths = [
    'processed_master_datasets/master_dataset_0.5s.npz',
    'processed_master_datasets/master_dataset_1.0s.npz',
    'processed_master_datasets/master_dataset_2.0s.npz',
    'processed_master_datasets/master_dataset_4.0s.npz',
    'processed_master_datasets/master_dataset_5.0s.npz',
    'processed_master_datasets/master_dataset_10.0s.npz'
]


# %% Pure CNN Model Definition
def build_seizure_model_cnn(input_shape, learning_rate=0.001, threshold=0.3):
    """
    Pure CNN model optimized for throughput.
    GlobalAveragePooling1D ensures parameter count stays constant regardless of window size.
    """
    inputs = layers.Input(shape=input_shape, name='eeg_input')

    # Block 1
    x = layers.Conv1D(32, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv1D(64, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv1D(128, kernel_size=7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 4 - Deep feature extraction
    x = layers.Conv1D(256, kernel_size=7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Pooling - KEEPS PARAMETERS CONSTANT ACROSS DIFFERENT WINDOW SIZES
    x = layers.GlobalAveragePooling1D()(x)

    # Classification Head
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)

    # Output MUST be float32 when using mixed precision
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='seizure_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='SeizureDetector_PureCNN')

    # Optimizer fixed learning rate (No ReduceLROnPlateau will be used)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=threshold),
            tf.keras.metrics.AUC(name='auc'),  # AUC is threshold-agnostic
            tf.keras.metrics.Precision(name='precision', thresholds=threshold),
            tf.keras.metrics.Recall(name='recall', thresholds=threshold)
        ]
    )
    return model

# ==========================================
# %% Main Execution Loop Over All Datasets
# ==========================================

for dataset_path in dataset_paths:
    print("\n" + "="*60)
    print(f"🚀 STARTING PROCESSING FOR: {dataset_path}")
    print("="*60)

    # Dynamically extract suffix and create unique timestamp/folders for this dataset
    file_name = os.path.splitext(os.path.basename(dataset_path))[0]
    suffix = file_name.split('_')[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("saved_outputs_play", f"{timestamp}_{suffix}")

    models_dir = os.path.join(output_dir, "models")
    histories_dir = os.path.join(output_dir, "histories")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(histories_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # %% Data Loading
    print(f"Loading data from {dataset_path}...")

    data = np.load(dataset_path)
    X, y, groups = data['X'], data['y'], data['s']
    X = np.swapaxes(X, 0, 1)

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
        print(f"\n🚀 Training holding out Subject: {current_test_subject} (Dataset: {suffix})")

        # Subject-level Validation Split
        unique_train_subjects = np.unique(groups_train)
        val_subject = unique_train_subjects[0]
        val_mask = (groups_train == val_subject)
        train_mask = ~val_mask

        X_train, X_val = X_train_full[train_mask], X_train_full[val_mask]
        y_train, y_val = y_train_full[train_mask], y_train_full[val_mask]

        # =========================================================================
        # DYNAMIC BALANCING FOR TRAINING ONLY (OOM FIXED)
        # =========================================================================

        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]

        X_train_pos, y_train_pos = X_train[pos_idx], y_train[pos_idx]
        X_train_neg, y_train_neg = X_train[neg_idx], y_train[neg_idx]

        print(f"   -> Training Split: {len(pos_idx)} Seizures, {len(neg_idx)} Normal")

        # Force the creation of the dataset graph onto the CPU to save GPU VRAM
        with tf.device('/CPU:0'):
            # Pass the arrays as they are, DO NOT use .astype() here
            pos_ds = tf.data.Dataset.from_tensor_slices((X_train_pos, y_train_pos))
            neg_ds = tf.data.Dataset.from_tensor_slices((X_train_neg, y_train_neg))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

        # Cast types dynamically inside the pipeline using .map()
        def cast_to_float32(x, y):
            return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

        pos_ds = pos_ds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        if len(pos_idx) > 0:
            pos_ds = pos_ds.shuffle(len(pos_idx))
        pos_ds = pos_ds.repeat()

        neg_ds = neg_ds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        if len(neg_idx) > 0:
            neg_ds = neg_ds.shuffle(len(neg_idx))
        neg_ds = neg_ds.repeat()

        balanced_train_ds = tf.data.Dataset.sample_from_datasets(
            [pos_ds, neg_ds],
            weights=[0.3, 0.7]
        )

        if len(pos_idx) > 0:
            STEPS_PER_EPOCH = max(1, (2 * len(pos_idx)) // BATCH_SIZE)
        else:
            STEPS_PER_EPOCH = 100

        train_dataset = balanced_train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Validation dataset (Sequential, imbalanced, dynamically cast)
        val_dataset = val_dataset.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        # =========================================================================

        # Build model with fixed learning rate
        model = build_seizure_model_cnn(input_shape, learning_rate=LEARNING_RATE, threshold=DECISION_THRESHOLD)

        fold_log_dir = os.path.join(log_dir, f"subject_{current_test_subject}")
        model_save_path = os.path.join(models_dir, f"best_model_subject_{current_test_subject}.keras")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
            TensorBoard(log_dir=fold_log_dir, histogram_freq=1),
            ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=0)
        ]

        # Train using the balanced tf.data pipeline
        history = model.fit(
            train_dataset,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_csv_path = os.path.join(histories_dir, f"history_subject_{current_test_subject}.csv")
        history_df.to_csv(history_csv_path, index=False)
        print(f"✅ Saved history for subject {current_test_subject} (Dataset {suffix})")

        # ==========================================
        # 🧹 SENIOR DS MEMORY CLEANUP (Inner Loop)
        # ==========================================
        del X_train_full, X_test, y_train_full, y_test
        del X_train, X_val, y_train, y_val
        del train_dataset, val_dataset
        del pos_ds, neg_ds, balanced_train_ds
        del model, history

        K.clear_session()
        gc.collect()

    # ==========================================
    # 🧹 MASTER MEMORY CLEANUP (Outer Loop)
    # ==========================================
    # Veri seti değiştirmeden önce, eski veri setini tamamen RAM'den siliyoruz!
    print(f"🧹 Clearing RAM before loading the next dataset...")
    del X, y, groups, data
    gc.collect()
    print(f"🎉 Training complete for dataset: {dataset_path}\n")

print("\n🏆 ALL DATASETS PROCESSED SUCCESSFULLY! 🏆")