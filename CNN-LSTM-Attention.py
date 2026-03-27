# %% Imports and Configuration
import os
import time
import datetime
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow.keras.backend as K

# [FIXED] Disable internal HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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

    def __init__(self, name='balanced_accuracy', threshold=0.50, **kwargs):
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
    """

    def __init__(self, filepath, monitor='val_balanced_accuracy', mode='max', verbose=0):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_metric = -np.inf if mode == 'max' else np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if current_metric is not None:
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
LEARNING_RATE = 1e-4
EPOCHS = 100
DECISION_THRESHOLD = 0.50

dataset_paths = [
    'processed_master_datasets/master_dataset_0.5s.npz',
    'processed_master_datasets/master_dataset_1.0s.npz',
    'processed_master_datasets/master_dataset_2.0s.npz',
    'processed_master_datasets/master_dataset_4.0s.npz',
    'processed_master_datasets/master_dataset_5.0s.npz',
    'processed_master_datasets/master_dataset_10.0s.npz'
]


# ==========================================
# %% CNN-LSTM-Attention Model Definition
# ==========================================
def create_model_cnn_lstm_attention(input_shape, learning_rate=1e-4, threshold=0.50):
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
    conv1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.MaxPooling1D(pool_size=2)(conv1)
    conv1 = layers.Dropout(0.2)(conv1)

    conv2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.MaxPooling1D(pool_size=2)(conv2)
    conv2 = layers.Dropout(0.2)(conv2)

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

    outputs = layers.Dense(1, activation='sigmoid', name='seizure_output')(dense2)

    model = Model(inputs=inputs, outputs=outputs, name='SeizureDetector_CNNLSTMAttention')

    # Using Focal Loss to handle extreme class imbalance
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0, from_logits=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.AUC(name='auc', curve='PR'),
            BalancedAccuracy(name='balanced_accuracy', threshold=threshold),
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

    if not os.path.exists(dataset_path):
        print(f"⚠️ Warning: Dataset not found at {dataset_path}. Skipping...")
        continue

    file_name = os.path.splitext(os.path.basename(dataset_path))[0]
    suffix = file_name.split('_')[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("saved_outputs_hybrid", f"{timestamp}_{suffix}_CNN_LSTM_ATTENTION")

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

    # Swap axes if required based on previous structure
    if len(X.shape) == 3 and X.shape[0] < X.shape[1]:
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

        # Validation Split - Rastgele %10 Hasta
        unique_train_subjects = np.unique(groups_train)
        num_val_subjects = max(1, min(3, int(len(unique_train_subjects) * 0.10)))

        np.random.seed(42)
        val_subjects = np.random.choice(unique_train_subjects, size=num_val_subjects, replace=False)

        val_mask = np.isin(groups_train, val_subjects)
        train_mask = ~val_mask

        X_train, X_val = X_train_full[train_mask], X_train_full[val_mask]
        y_train, y_val = y_train_full[train_mask], y_train_full[val_mask]

        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]

        # -------------------------------------------------------------
        # [CRITICAL FIX] 0 Nöbet Koruması (Sonsuz Döngü / Çökme Önleyici)
        # -------------------------------------------------------------
        if len(pos_idx) == 0:
            print(f"⚠️ DİKKAT: Fold {current_test_subject} için ayrılan eğitim setinde hiç nöbet verisi yok!")
            print("Bu fold modeli yanıltmamak adına atlanıyor...")
            continue

        X_train_pos, y_train_pos = X_train[pos_idx], y_train[pos_idx]
        X_train_neg, y_train_neg = X_train[neg_idx], y_train[neg_idx]

        print(f"   -> Training Split: {len(pos_idx)} Seizures, {len(neg_idx)} Normal")
        print(f"   -> Validation Split based on subjects: {val_subjects}")

        with tf.device('/CPU:0'):
            pos_ds = tf.data.Dataset.from_tensor_slices((X_train_pos, y_train_pos))
            neg_ds = tf.data.Dataset.from_tensor_slices((X_train_neg, y_train_neg))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))


        def cast_to_float32(x, target_y):
            return tf.cast(x, tf.float32), tf.cast(target_y, tf.float32)


        pos_ds = pos_ds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        if len(pos_idx) > 0:
            pos_ds = pos_ds.shuffle(len(pos_idx))
        pos_ds = pos_ds.repeat()

        neg_ds = neg_ds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        if len(neg_idx) > 0:
            neg_ds = neg_ds.shuffle(len(neg_idx))
        neg_ds = neg_ds.repeat()

        # Dengeleme: Tam %50 - %50
        balanced_train_ds = tf.data.Dataset.sample_from_datasets(
            [pos_ds, neg_ds],
            weights=[0.5, 0.5]
        )

        STEPS_PER_EPOCH = max(1, (2 * len(pos_idx)) // BATCH_SIZE)

        train_dataset = balanced_train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Validation Dataset pipeline
        val_dataset = val_dataset.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        with strategy.scope():
            model = create_model_cnn_lstm_attention(
                input_shape,
                learning_rate=LEARNING_RATE,
                threshold=DECISION_THRESHOLD
            )

        fold_log_dir = os.path.join(log_dir, f"subject_{current_test_subject}")
        model_save_path = os.path.abspath(os.path.join(models_dir, f"best_model_subject_{current_test_subject}.keras"))

        callbacks = [
            EarlyStopping(monitor='val_balanced_accuracy', mode='max', patience=25, restore_best_weights=True,
                          verbose=1),
            TensorBoard(log_dir=fold_log_dir, histogram_freq=0),
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

        history_df = pd.DataFrame(history.history)
        history_csv_path = os.path.join(histories_dir, f"history_subject_{current_test_subject}.csv")
        history_df.to_csv(history_csv_path, index=False)
        print(f"✅ Saved history for subject {current_test_subject} (Dataset {suffix})")

        # Memory Cleanup
        del model, train_dataset, val_dataset, balanced_train_ds, pos_ds, neg_ds
        del X_train_full, X_test, y_train_full, y_test, X_train, X_val, y_train, y_val
        K.clear_session()
        gc.collect()

    # Clear large arrays before moving to the next dataset resolution
    del data, X, y, groups
    gc.collect()
    print(f"🎉 Training complete for dataset: {dataset_path}\n")

print("\n🏆 ALL DATASETS PROCESSED SUCCESSFULLY! 🏆")