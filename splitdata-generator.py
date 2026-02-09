import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# ============================================
# 1. DATA SPLITTING (FILE LEVEL)
# ============================================
print("=" * 60)
print("SPLITTING DATA (FILE LEVEL)")
print("=" * 60)

# 1. Get all .npz files
PROCESSED_DIR = 'processed_dataset_2'  # Ensure this matches your folder name
all_files = glob.glob(os.path.join(PROCESSED_DIR, "*.npz"))

# 2. Extract Recording IDs (e.g., 'chb01_03') to prevent leakage
# Format: chb01_03_P1.npz -> ID: chb01_03
file_ids = [os.path.basename(f).split('_P')[0] for f in all_files]
unique_ids = list(set(file_ids))

# 3. Split IDs (70% Train, 15% Val, 15% Test)
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)


# 4. Filter file paths based on split IDs
def filter_files(file_list, target_ids):
    return [f for f in file_list if os.path.basename(f).split('_P')[0] in target_ids]


train_files = filter_files(all_files, train_ids)
val_files = filter_files(all_files, val_ids)
test_files = filter_files(all_files, test_ids)


# --- Helper to count stats (Optional but good for verification) ---
def get_set_stats(file_list):
    total_samples = 0
    seizure_samples = 0
    # Scanning first 50 files to save time, or scan all if dataset is small
    for f in file_list:
        try:
            with np.load(f) as data:
                y = data['y']
                total_samples += len(y)
                seizure_samples += np.sum(y)
        except:
            pass
    return total_samples, int(seizure_samples)


print(f"\n File Counts:")
print(f"   • Train Files: {len(train_files)}")
print(f"   • Val Files:   {len(val_files)}")
print(f"   • Test Files:  {len(test_files)}")
print("\n" + "=" * 60)


# ============================================
# 2. FUNCTIONAL DATA GENERATOR (tf.data)
# ============================================

def load_file_data(file_path):
    """
    Python function to load a single .npz file and preprocess it.
    """
    # 1. Convert Tensor string to normal Python string
    path = file_path.numpy().decode('utf-8')

    # 2. Load Data
    data = np.load(path)
    x = data['x']  # Shape: (N_epochs, 18, 256)
    y = data['y']  # Shape: (N_epochs,)

    # 3. Transpose for Conv1D: (N, Channels, Time) -> (N, Time, Channels)
    # Target Shape: (N, 256, 18)
    x = np.transpose(x, (0, 2, 1))

    # 4. Ensure correct data types (Float32 for input, Int/Float for label)
    return x.astype(np.float32), y.astype(np.float32)


def create_dataset(file_paths, batch_size=32, shuffle=True):
    """
    Creates a highly optimized TensorFlow Dataset pipeline.
    """
    # 1. Create a dataset of file paths
    ds = tf.data.Dataset.from_tensor_slices(file_paths)

    if shuffle:
        # Shuffle the order of files first
        ds = ds.shuffle(buffer_size=len(file_paths))

    # 2. Map the loading function (using tf.py_function because np.load is Python code)
    # We define the output shapes explicitly
    ds = ds.map(
        lambda x: tf.py_function(load_file_data, [x], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 3. Fix Shapes (tf.py_function loses shape info, we must restore it)
    # Shape: (None, 256, 18) -> None means unknown number of epochs per file
    def set_shapes(x, y):
        x.set_shape([None, 256, 18])
        y.set_shape([None])
        return x, y

    ds = ds.map(set_shapes)

    # 4. Unbatch: Turn "Dataset of Files" into "Dataset of Epochs"
    # This flattens the stream so we can batch individual samples, not whole files
    ds = ds.unbatch()

    # 5. Shuffle individual epochs (Crucial for training stability)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)  # Buffer size controls randomness

    # 6. Create Mini-Batches
    ds = ds.batch(batch_size)

    # 7. Prefetch (Preload next batch while GPU processes current one)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ============================================
# 3. CREATE DATASETS & TRAIN
# ============================================

# Create the functional datasets
# Note: batch_size=64 here means 64 epochs (samples), NOT 64 files.
# This provides much more stable gradients than the file-based batching.
train_ds = create_dataset(train_files, batch_size=64, shuffle=True)
val_ds = create_dataset(val_files, batch_size=64, shuffle=False)
test_ds = create_dataset(test_files, batch_size=64, shuffle=False)

print("Data Pipelines Created Successfully")

# --- Model Training Example ---
# model.fit(train_ds, validation_data=val_ds, epochs=10)