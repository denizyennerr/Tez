# %% Imports and Configuration
import os
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
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
# %% Kolmogorov-Arnold Network (KAN) Layers
# ==========================================
# These are self-contained Keras implementations of the B-spline KAN
# formulation (Liu et al. 2024 / "efficient-kan"). Each connection learns its
# own activation function as a linear combination of B-spline bases plus a
# residual SiLU branch:
#
#     y = SiLU(x) @ W_base  +  Bspline(x) @ W_spline
#
# This replaces the fixed (Conv + ReLU) units of the CNN with learnable
# univariate activations, so the network here is "fully convolutional KAN":
# KAN convolutions for feature extraction and KAN dense layers for the head.

@tf.keras.utils.register_keras_serializable(package="KAN")
class KANLinear(layers.Layer):
    """A KAN dense layer. Expects 2-D input of shape (batch, in_features)."""

    def __init__(self, units, grid_size=5, spline_order=3,
                 grid_range=(-1.0, 1.0), spline_init_std=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = tuple(grid_range)
        self.spline_init_std = float(spline_init_std)

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        self.in_features = in_features
        # Number of B-spline bases produced per input feature.
        self.num_bases = self.grid_size + self.spline_order

        # Fixed uniform knot vector, shared across all input features.
        # Length = grid_size + 2 * spline_order + 1
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        knots = np.arange(-self.spline_order,
                          self.grid_size + self.spline_order + 1,
                          dtype=np.float32) * h + self.grid_range[0]
        grid = np.tile(knots[None, :], (in_features, 1))  # (in_features, G)
        self.grid = tf.constant(grid, dtype=tf.float32)

        # Residual (base) branch weights: SiLU(x) @ W_base
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(in_features, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        # Spline branch weights: flattened (in_features * num_bases) -> units
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(in_features * self.num_bases, self.units),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.spline_init_std),
            trainable=True,
        )
        super().build(input_shape)

    def b_splines(self, x):
        """Evaluate B-spline bases. x: (batch, in_features) -> (batch, in_features, num_bases)."""
        x = tf.expand_dims(x, axis=-1)  # (batch, in_features, 1)
        grid = self.grid                # (in_features, G)

        # Order-0 (piecewise constant) bases.
        bases = tf.cast((x >= grid[:, :-1]) & (x < grid[:, 1:]), x.dtype)

        # Cox-de Boor recursion up to the requested spline order.
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases  # (batch, in_features, num_bases)

    def call(self, inputs):
        base_out = tf.matmul(tf.nn.silu(inputs), self.base_weight)
        bs = self.b_splines(inputs)
        bs_flat = tf.reshape(bs, (-1, self.in_features * self.num_bases))
        spline_out = tf.matmul(bs_flat, self.spline_weight)
        return base_out + spline_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "spline_init_std": self.spline_init_std,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="KAN")
class KANConv1D(layers.Layer):
    """1-D KAN convolution. Slides a shared KAN over (kernel_size * channels) patches."""

    def __init__(self, filters, kernel_size, strides=1, padding="SAME",
                 grid_size=5, spline_order=3, grid_range=(-1.0, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = padding.upper()
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = tuple(grid_range)

    def build(self, input_shape):
        self.in_channels = int(input_shape[-1])
        # A single KAN dense layer shared across every spatial position. Its
        # input is the flattened receptive field (kernel_size * in_channels).
        self.kan = KANLinear(
            self.filters,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            grid_range=self.grid_range,
            name="kan_kernel",
        )
        self.kan.build((None, self.kernel_size * self.in_channels))
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, length, channels). Treat as a width-1 image so we can
        # use extract_patches to gather sliding receptive fields.
        x = tf.expand_dims(inputs, axis=2)  # (batch, length, 1, channels)
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, 1, 1],
            strides=[1, self.strides, 1, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding,
        )  # (batch, out_len, 1, kernel_size * channels)

        b = tf.shape(patches)[0]
        out_len = tf.shape(patches)[1]
        patch_dim = self.kernel_size * self.in_channels

        patches = tf.reshape(patches, (b * out_len, patch_dim))
        out = self.kan(patches)                       # (b * out_len, filters)
        out = tf.reshape(out, (b, out_len, self.filters))
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
        })
        return config


# ==========================================
# %% Custom Metric for Balanced Accuracy
# ==========================================
@tf.keras.utils.register_keras_serializable(package="KAN")
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

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


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
LEARNING_RATE = 0.0001
EPOCHS = 100
DECISION_THRESHOLD = 0.50

# KAN hyper-parameters (shared by conv and dense KAN layers)
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3

dataset_paths = [
    'processed_master_datasets/master_dataset_0.5s.npz',
    'processed_master_datasets/master_dataset_1.0s.npz',
    'processed_master_datasets/master_dataset_2.0s.npz',
    'processed_master_datasets/master_dataset_4.0s.npz',
    'processed_master_datasets/master_dataset_5.0s.npz',
    'processed_master_datasets/master_dataset_10.0s.npz'
]

# Fall back to whatever datasets are actually present on disk.
dataset_paths = [p for p in dataset_paths if os.path.exists(p)]
if not dataset_paths:
    for candidate in ('master_dataset_0.5s.npz', 'processed_master_datasets'):
        if os.path.isfile(candidate):
            dataset_paths = [candidate]
            break
    else:
        # Pick up any local master_dataset_*.npz files.
        dataset_paths = sorted(
            f for f in os.listdir('.')
            if f.startswith('master_dataset_') and f.endswith('.npz')
        )


def load_dataset(path):
    """Load X, y, groups ensuring X is (samples, time, channels)."""
    data = np.load(path)
    X, y, groups = data['X'], data['y'], data['s']
    n = len(y)
    if X.shape[0] != n:
        # Move the sample axis to the front if it sits elsewhere.
        sample_axes = [i for i, s in enumerate(X.shape) if s == n]
        if sample_axes:
            X = np.moveaxis(X, sample_axes[0], 0)
    return X, y, groups


# %% Fully Convolutional KAN Model Definition
def build_seizure_model_kan(input_shape, learning_rate=0.001, threshold=0.50,
                            dropout_factor=0.25, grid_size=KAN_GRID_SIZE,
                            spline_order=KAN_SPLINE_ORDER):
    inputs = layers.Input(shape=input_shape, name='eeg_input')

    kan_kwargs = dict(grid_size=grid_size, spline_order=spline_order)

    # Block 1
    x = KANConv1D(32, kernel_size=3, padding='same', **kan_kwargs)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2 * dropout_factor)(x)

    # Block 2
    x = KANConv1D(64, kernel_size=5, padding='same', **kan_kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2 * dropout_factor)(x)

    # Block 3
    x = KANConv1D(128, kernel_size=7, padding='same', **kan_kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3 * dropout_factor)(x)

    # Block 4
    x = KANConv1D(256, kernel_size=7, padding='same', **kan_kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3 * dropout_factor)(x)

    # Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # KAN Classification Head
    x = KANLinear(64, **kan_kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5 * dropout_factor)(x)

    logits = KANLinear(1, **kan_kwargs)(x)
    outputs = layers.Activation('sigmoid', dtype='float32', name='seizure_output')(logits)

    model = Model(inputs=inputs, outputs=outputs, name='SeizureDetector_FullKAN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=threshold),
            BalancedAccuracy(name='balanced_accuracy', threshold=threshold),
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

    file_name = os.path.splitext(os.path.basename(dataset_path))[0]
    suffix = file_name.split('_')[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("saved_outputs_kan", f"{timestamp}_{suffix}")

    models_dir = os.path.join(output_dir, "models")
    histories_dir = os.path.join(output_dir, "histories")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(histories_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # %% Data Loading
    print(f"Loading data from {dataset_path}...")

    X, y, groups = load_dataset(dataset_path)

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
            model = build_seizure_model_kan(
                input_shape,
                learning_rate=LEARNING_RATE,
                threshold=DECISION_THRESHOLD,
                dropout_factor=0.25,
                grid_size=KAN_GRID_SIZE,
                spline_order=KAN_SPLINE_ORDER
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

        K.clear_session()

    print(f"🎉 Training complete for dataset: {dataset_path}\n")

print("\n🏆 ALL DATASETS PROCESSED SUCCESSFULLY! 🏆")
