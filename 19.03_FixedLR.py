# %% Imports and Configuration
import os

# [FIXED] Disable internal HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import time
import datetime
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow.keras.backend as K

# Check for GPUs and set up MirroredStrategy for Multi-GPU training
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"✅ GPUs Detected: {len(physical_devices)}")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    print(f"🚀 Training distributed across {strategy.num_replicas_in_sync} GPUs")
else:
    print("⚠️ WARNING: No GPU detected. Running on CPU.")
    strategy = tf.distribute.get_strategy()


# ==========================================
# %% Custom Metric for Balanced Accuracy
# ==========================================
class BalancedAccuracy(tf.keras.metrics.Metric):
    """
    Computes Balanced Accuracy: (Sensitivity + Specificity) / 2
    """

    def __init__(self, name='balanced_accuracy', threshold=0.3, **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.tn.assign_add(tf.reduce_sum((1 - y_true) * (1 - y_pred)))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        sensitivity = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        specificity = tf.math.divide_no_nan(self.tn, self.tn + self.fp)
        return (sensitivity + specificity) / 2.0

    def reset_state(self):
        self.tp.assign(0.0)
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


# ==========================================
# %% Custom Callback for Windows File Locking
# ==========================================
class SafeModelCheckpoint(Callback):
    """
    A custom checkpoint callback that catches Windows PermissionErrors.
    Updated to handle 'max' or 'min' modes for metric monitoring.
    """

    def __init__(self, filepath, monitor='val_balanced_accuracy', mode='max', verbose=0):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        # If maximizing, start at negative infinity
        self.best_metric = -np.inf if mode == 'max' else np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if current_metric is not None:
            # Check improvement based on mode
            improved = (current_metric > self.best_metric) if self.mode == 'max' else (
                        current_metric < self.best_metric)

            if improved:
                if self.verbose > 0:
                    print(
                        f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best_metric:.5f} to {current_metric:.5f}, saving model...")
                self.best_metric = current_metric

                for attempt in range(5):
                    try:
                        self.model.save(self.filepath)
                        break
                    except PermissionError:
                        print(f"\n⚠️ [Attempt {attempt + 1}/5] Windows locked the file. Retrying in 2 seconds...")
                        time.sleep(2)
                else:
                    print(
                        f"\n❌ Failed to save model to {self.filepath} after 5 attempts due to persistent PermissionError.")


# ==========================================
# %% Constants & Dataset Paths
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 100
DECISION_THRESHOLD = 0.30

dataset_paths = [
    'processed_master_datasets/master_dataset_0.5s.npz',
    'processed_master_datasets/master_dataset_1.0s.npz',
    'processed_master_datasets/master_dataset_2.0s.npz',
    'processed_master_datasets/master_dataset_4.0s.npz',
    'processed_master_datasets/master_dataset_5.0s.npz',
    'processed_master_datasets/master_dataset_10.0s.npz'
]


# %% Pure CNN Model Definition
def build_seizure_model_cnn(input_shape, learning_rate=0.001, threshold=0.3, dropout_factor=0.5):
    """
    Pure CNN model.
    Dropout rates are scaled by dropout_factor to easily lower dropout globally without changing the architecture.
    """
    inputs = layers.Input(shape=input_shape, name='eeg_input')

    # Block 1
    x = layers.Conv1D(32, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2 * dropout_factor)(x)

    # Block 2
    x = layers.Conv1D(64, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2 * dropout_factor)(x)

    # Block 3
    x = layers.Conv1D(128, kernel_size=7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3 * dropout_factor)(x)

    # Block 4 - Deep feature extraction
    x = layers.Conv1D(256, kernel_size=7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3 * dropout_factor)(x)

    # Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classification Head
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5 * dropout_factor)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3 * dropout_factor)(x)

    # Output MUST be float32 when using mixed precision
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='seizure_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='SeizureDetector_PureCNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=threshold),
            BalancedAccuracy(name='balanced_accuracy', threshold=threshold),  # Added Custom Metric
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision', thresholds=threshold),
            tf.keras.metrics.Recall(name='recall', thresholds=threshold)
        ]
    )
    return model


# ==========================================
# %% Main Execution Loop Over All Datasets
# ==========================================

for dataset_path in dataset_paths:
    print("\n" + "=" * 60)
    print(f"🚀 STARTING PROCESSING FOR: {dataset_path}")
    print("=" * 60)

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

        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]

        X_train_pos, y_train_pos = X_train[pos_idx], y_train[pos_idx]
        X_train_neg, y_train_neg = X_train[neg_idx], y_train[neg_idx]

        print(f"   -> Training Split: {len(pos_idx)} Seizures, {len(neg_idx)} Normal")

        with tf.device('/CPU:0'):
            pos_ds = tf.data.Dataset.from_tensor_slices((X_train_pos, y_train_pos))
            neg_ds = tf.data.Dataset.from_tensor_slices((X_train_neg, y_train_neg))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))


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

        val_dataset = val_dataset.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        with strategy.scope():
            # I set dropout_factor to 0.5 to cut all dropouts in half
            model = build_seizure_model_cnn(
                input_shape,
                learning_rate=LEARNING_RATE,
                threshold=DECISION_THRESHOLD,
                dropout_factor=0.5
            )

        fold_log_dir = os.path.join(log_dir, f"subject_{current_test_subject}")
        model_save_path = os.path.abspath(os.path.join(models_dir, f"best_model_subject_{current_test_subject}.keras"))

        callbacks = [
            # Swapped EarlyStopping to monitor val_balanced_accuracy and mode='max'
            EarlyStopping(monitor='val_balanced_accuracy', mode='max', patience=25, restore_best_weights=True,
                          verbose=1),
            TensorBoard(log_dir=fold_log_dir, histogram_freq=1),
            SafeModelCheckpoint(filepath=model_save_path, monitor='val_auc', mode='max', verbose=0)
        ]

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

        del X_train_full, X_test, y_train_full, y_test
        del X_train, X_val, y_train, y_val
        del train_dataset, val_dataset
        del pos_ds, neg_ds, balanced_train_ds
        del model, history

        K.clear_session()
        gc.collect()

    print(f"🧹 Clearing RAM before loading the next dataset...")
    del X, y, groups, data
    gc.collect()
    print(f"🎉 Training complete for dataset: {dataset_path}\n")

print("\n🏆 ALL DATASETS PROCESSED SUCCESSFULLY! 🏆")