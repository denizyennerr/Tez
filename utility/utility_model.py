import os
import glob
import numpy as np
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import layers, models


def report_folder_sizes(target_path):
    # List all items in the specified directory
    try:
        content = os.listdir(target_path)
    except FileNotFoundError:
        print("Error: The specified directory was not found.")
        return
    except PermissionError:
        print("Error: Permission denied.")
        return

    # Print Header
    print(f"{'Folder Name':<40} | {'Size (MB)':<12}")
    print("-" * 55)

    for item in content:
        full_path = os.path.join(target_path, item)

        # Check if the item is a directory
        if os.path.isdir(full_path):
            total_size = 0

            # Walk through all subdirectories and files
            for root, dirs, files in os.walk(full_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file size if the file exists (prevents issues with broken symlinks)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)

            # Convert Bytes to Megabytes (1 MB = 1024 * 1024 Bytes)
            size_mb = total_size / (1024 * 1024)

            print(f"{item:<40} | {size_mb:>10.2f} MB")


def extract_subject(filename):
    return filename.split("_")[0]


def get_npz_index(dataset_root):
    """
    Dataset altındaki train ve val klasörlerini tarayarak subject bazlı bir indeks oluşturur.
    Yapı: dataset_root/{train|val}/{subject}/*.npz
    """
    index = defaultdict(lambda: {"train": [], "val": []})

    def fill_index(split_name):
        split_path = os.path.join(dataset_root, split_name)

        if not os.path.exists(split_path):
            print(f"Uyarı: {split_path} dizini bulunamadı.")
            return

        # Subject klasörlerini al (chb01, chb02 vb.)
        subjects = [s for s in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, s))]

        for subject in subjects:
            subject_path = os.path.join(split_path, subject)
            # .npz uzantılı tüm dosyaların tam yolunu bul
            files = glob.glob(os.path.join(subject_path, "*.npz"))
            index[subject][split_name].extend(sorted(files))

    # Hem train hem val için işlemi çalıştır
    fill_index("train")
    fill_index("val")

    return dict(index)


def loso_split(index_dict, test_subjects):
    if isinstance(test_subjects, str):
        test_subjects = [test_subjects]

    test_subjects = set(test_subjects)

    train_files = []
    val_files = []

    for subject, splits in index_dict.items():

        if subject in test_subjects:
            val_files.extend(splits.get("val", []))
        else:
            train_files.extend(splits.get("train", []))

    return train_files, val_files


def npz_file_generator(file_list):
    for f in file_list:
        data = np.load(f)

        X = data["X"]
        y = data["y"]

        yield X, y


# # Inside batch_generator, change the yield block to this:
# if len(X_buffer) == batch_size:
#     X_batch = np.array(X_buffer)
#     y_batch = np.array(y_buffer)
#
#     # Shuffle X and y IN UNISON to mix classes
#     indices = np.arange(batch_size)
#     np.random.shuffle(indices)
#
#     yield X_batch[indices], y_batch[indices]
#
#     X_buffer, y_buffer = [], []


def robust_batch_generator(file_list, batch_size=32, shuffle=True):
    X_buffer = []
    y_buffer = []

    while True:
        if shuffle:
            np.random.shuffle(file_list)

        # Assuming npz_file_generator is defined in your utils
        for X, y in npz_file_generator(file_list):

            for i in range(len(X)):
                X_buffer.append(X[i])
                y_buffer.append(y[i])

                # When buffer is full...
                if len(X_buffer) >= batch_size:
                    X_batch = np.array(X_buffer[:batch_size])
                    y_batch = np.array(y_buffer[:batch_size])

                    # Clear processed items
                    X_buffer = X_buffer[batch_size:]
                    y_buffer = y_buffer[batch_size:]

                    # --- CRITICAL FIX: SHUFFLE THE BATCH ---
                    # This mixes 1s and 0s so the model sees both classes at once
                    if shuffle:
                        idx = np.arange(len(X_batch))
                        np.random.shuffle(idx)
                        X_batch = X_batch[idx]
                        y_batch = y_batch[idx]

                    yield X_batch, y_batch


def batch_generator(file_list, batch_size=32, shuffle=True):
    X_buffer = []
    y_buffer = []

    while True:

        files = file_list.copy()
        if shuffle:
            np.random.shuffle(files)

        for X, y in npz_file_generator(files):

            for i in range(len(X)):
                X_buffer.append(X[i])
                y_buffer.append(y[i])

                if len(X_buffer) == batch_size:
                    yield np.array(X_buffer), np.array(y_buffer)
                    X_buffer, y_buffer = [], []


def compute_zscore_stats(train_files):
    total_sum = 0
    total_sq = 0
    total_count = 0

    for X, _ in npz_file_generator(train_files):
        total_sum += X.sum(axis=(0, 2))
        total_sq += (X ** 2).sum(axis=(0, 2))
        total_count += X.shape[0] * X.shape[2]

    mean = total_sum / total_count
    std = np.sqrt(total_sq / total_count - mean ** 2)

    return mean, std


def apply_zscore(X, mean, std):
    return (X - mean[:, None]) / (std[:, None] + 1e-8)


def normalized_batch_generator(file_list, mean, std, batch_size=32):
    for X, y in batch_generator(file_list, batch_size):
        X = apply_zscore(X, mean, std)
        yield X, y


def normalized_batch_generator_v2(file_list, mean, std, batch_size=32):
    for X, y in robust_batch_generator(file_list, batch_size):
        X = apply_zscore(X, mean, std)
        yield X, y


def build_tuned_model():
    model = build_cnn_model()

    # Use 1e-4 instead of default 1e-3
    # Clipnorm prevents exploding gradients from EEG artifacts
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_model_v2(n_channels=18, n_samples=256):
    """Lightweight CNN with heavy regularization for small EEG datasets"""
    inputs = layers.Input(shape=(n_channels, n_samples))
    x = layers.Permute((2, 1))(inputs)  # (samples, channels) -> (time, channels)

    # Block 1: Gentle feature extraction
    x = layers.Conv1D(16, kernel_size=7, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)  # Aggressive pooling reduces overfitting
    x = layers.Dropout(0.4)(x)  # High dropout for regularization

    # Block 2: Minimal additional capacity
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.4)(x)

    # Global context instead of dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)  # Critical final dropout

    outputs = layers.Dense(1, activation='sigmoid',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    model = tf.keras.Model(inputs, outputs)

    # Conservative optimizer settings for EEG
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=5e-4,  # Slower learning = more stable
            clipnorm=1.0  # Prevent gradient explosions from artifacts
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', multi_label=False),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


def build_cnn_model(n_channels=18, n_samples=256, lr=0.001):
    inputs = layers.Input(shape=(n_channels, n_samples))

    x = layers.Permute((2, 1))(inputs)

    x = layers.Conv1D(64, 7, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()]
    )

    return model