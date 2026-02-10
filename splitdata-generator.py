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

def build_cnn_model(input_shape):
    model = models.Sequential()

    # 1. Temporal Block
    model.add(layers.Conv1D(32, 64, activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    # 2. Spatial Block
    model.add(layers.Conv1D(64, 16, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    # 3. Global Block
    model.add(layers.Conv1D(128, 8, activation='relu', padding='same'))
    model.add(layers.GlobalAveragePooling1D())

    # Classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    return model


model = build_cnn_model(input_shape=(256, 18))
# model.summary()

# from torchsummary import summary
# imgSize = 32
# # count the total number of parameters in the model
# summary(net,(3,imgSize,imgSize))

# ============================================
# 3. TRAIN & PREDICT (The Fix)
# ============================================
print("\n🚀 Starting Training...")

# FIX: Use train_ds and val_ds (not train_gen)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
)

print("\n📊 Evaluating on Test Set...")
# Evaluate returns [loss, accuracy, recall]
test_results = model.evaluate(test_ds)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test Recall: {test_results[2]:.4f}")

print("\n🔮 Generating Predictions...")
# Get probabilities (0.0 to 1.0)
y_pred_probs = model.predict(test_ds)
# Convert to classes (0 or 1)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# To see True Labels, we must extract them from the dataset
y_true = np.concatenate([y for x, y in test_ds], axis=0)

print(f"Prediction Shape: {y_pred_classes.shape}")
print(f"True Labels Shape: {y_true.shape}")

# Optional: Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)


# Visualization
import matplotlib.pyplot as plt

# Optional: Set a clean style for professional-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

if 'history' in locals():
    h = history.history
    epochs = range(1, len(h['loss']) + 1)

    # Find the specific recall key names (e.g., 'recall', 'recall_1')
    rec_key = [k for k in h.keys() if 'recall' in k and 'val' not in k][0]
    val_rec_key = [k for k in h.keys() if 'val_recall' in k][0]

    # Create figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Loss Plot
    ax[0].plot(epochs, h['loss'], 'b-', label='Train Loss')
    ax[0].plot(epochs, h['val_loss'], 'r--', label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[0].grid(True)

    # 2. Accuracy Plot
    ax[1].plot(epochs, h['accuracy'], 'b-', label='Train Acc')
    ax[1].plot(epochs, h['val_accuracy'], 'r--', label='Val Acc')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    # 3. Recall Plot
    ax[2].plot(epochs, h[rec_key], 'b-', label='Train Recall')
    ax[2].plot(epochs, h[val_rec_key], 'r--', label='Val Recall')
    ax[2].set_title('Recall')
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.savefig('data-understanding/visualizations/results.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    print(" No history object found. Run model.fit() first.")