import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from collections import defaultdict
import random

# --- CONFIGURATION ---
DATASET_ROOT = "dataset"  # Root folder containing your .npz files (can have subfolders)
TEST_SUBJECT_ID = "chb16"  # The subject to hold out for testing
BATCH_SIZE = 64
EPOCH_LENGTH = 256
N_CHANNELS = 18


# --- 1. DATA INDEXING & SPLITTING ---

def get_subject_index(dataset_root):
    """
    Scans the folder recursively to find all .npz files and groups them by subject.
    Assumes filename format: "chb01_03_train.npz" -> Subject is "chb01"
    """
    subject_index = defaultdict(list)
    # Recursively find all .npz files
    all_files = glob.glob(os.path.join(dataset_root, "**", "*.npz"), recursive=True)

    for fpath in all_files:
        filename = os.path.basename(fpath)
        subject = filename.split('_')[0]  # Extracts 'chb01'
        subject_index[subject].append(fpath)

    print(f"Found {len(all_files)} files across {len(subject_index)} subjects.")
    return dict(subject_index)


def get_loso_split(subject_index, test_subject):
    """
    Leave-One-Subject-Out Split.
    Returns list of file paths for Train and Test.
    """
    train_files = []
    test_files = []

    if test_subject not in subject_index:
        raise ValueError(f"Subject {test_subject} not found in dataset index.")

    for subject, files in subject_index.items():
        if subject == test_subject:
            test_files.extend(files)
        else:
            train_files.extend(files)

    print(f"--- SPLIT SUMMARY ---")
    print(f"Test Subject: {test_subject}")
    print(f"Train Files:  {len(train_files)}")
    print(f"Test Files:   {len(test_files)}")

    return train_files, test_files


# --- 2. STATS CALCULATION (TRAIN ONLY) ---

def compute_global_stats(file_list):
    """
    Computes Mean and Std per channel across the training set.
    Assumes Data Shape in file: (N_epochs, Channels, Time)
    """
    print("Computing global stats on training data...")
    sum_x = np.zeros(N_CHANNELS)
    sum_sq_x = np.zeros(N_CHANNELS)
    total_count = 0

    # Process a subset if dataset is massive (e.g., random 200 files)
    sample_files = random.sample(file_list, min(len(file_list), 200))

    for fpath in sample_files:
        try:
            with np.load(fpath) as data:
                # Key 'x' is standard; check if your file uses 'X' or 'data'
                X = data['x'].astype(np.float32)

                # Sum across Epochs (0) and Time (2), keep Channels (1)
                sum_x += X.sum(axis=(0, 2))
                sum_sq_x += (X ** 2).sum(axis=(0, 2))
                total_count += X.shape[0] * X.shape[2]
        except Exception as e:
            print(f"Skipping corrupt file {fpath}: {e}")

    mean = sum_x / total_count
    std = np.sqrt((sum_sq_x / total_count) - (mean ** 2))

    # Reshape for broadcasting: (Channels, 1) to match (Channels, Time)
    return mean.reshape(-1, 1), std.reshape(-1, 1)


# --- 3. TF.DATA PIPELINE ---

def data_generator(file_list, mean, std):
    """
    Generator that loads files, normalizes, and yields single epochs.
    """
    # Cast constants to float32 for TF
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    for fpath in file_list:
        try:
            with np.load(fpath) as data:
                X = data['x'].astype(np.float32)  # Shape: (N, 18, Time)
                y = data['y'].astype(np.float32)  # Shape: (N,)

                # Z-Score Normalization
                X_norm = (X - mean) / (std + 1e-8)

                for i in range(len(X)):
                    yield X_norm[i], y[i]
        except:
            continue


def create_dataset(file_list, mean, std, is_train=True):
    """
    Creates a highly optimized tf.data.Dataset
    """
    if is_train:
        random.shuffle(file_list)

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_list, mean, std),
        output_signature=(
            tf.TensorSpec(shape=(N_CHANNELS, EPOCH_LENGTH), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    if is_train:
        dataset = dataset.shuffle(1000)  # Shuffle buffer

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Preload next batch
    return dataset

# --- 4. MODEL ARCHITECTURE ---

def build_eeg_cnn(input_shape=(18, 256)):
    inputs = layers.Input(shape=input_shape)

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

