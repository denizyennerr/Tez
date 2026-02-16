import os
import glob
import random
import numpy as np
import tensorflow as tf
from keras import layers, models, metrics
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib
import matplotlib.pyplot as plt
import warnings

matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")


def collect_npz_paths(root):
    return glob.glob(root + "/**/*.npz", recursive=True)


def load_npz_file(path):
    """Load .npz file with error handling"""
    try:
        data = np.load(path)
        X = data['X']  # (epochs, channels, time)
        y = data['y']
        X = np.transpose(X, (0, 2, 1))  # -> (epochs, time, channels)
        return X, y
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return np.array([]), np.array([])


def balanced_batch_generator(npz_paths, batch_size=32, buffer_size=4000, target_ratio=0.15):
    """
    ✅ FIXED: Handles files with insufficient samples
    """
    paths = npz_paths.copy()

    buffer_seizure_X = []
    buffer_seizure_y = []
    buffer_safe_X = []
    buffer_safe_y = []

    while True:
        random.shuffle(paths)

        for path in paths:
            try:
                X_file, y_file = load_npz_file(path)

                if len(X_file) == 0:
                    continue

                seizure_mask = (y_file == 1)
                safe_mask = (y_file == 0)

                if np.any(seizure_mask):
                    buffer_seizure_X.append(X_file[seizure_mask])
                    buffer_seizure_y.append(y_file[seizure_mask])

                if np.any(safe_mask):
                    buffer_safe_X.append(X_file[safe_mask])
                    buffer_safe_y.append(y_file[safe_mask])

                if len(buffer_seizure_X) > 0 and len(buffer_safe_X) > 0:
                    seizure_X = np.concatenate(buffer_seizure_X, axis=0)
                    seizure_y = np.concatenate(buffer_seizure_y, axis=0)
                    safe_X = np.concatenate(buffer_safe_X, axis=0)
                    safe_y = np.concatenate(buffer_safe_y, axis=0)

                    # Limit buffer size
                    if len(seizure_X) > buffer_size // 4:
                        seizure_X = seizure_X[-buffer_size // 4:]
                        seizure_y = seizure_y[-buffer_size // 4:]
                    if len(safe_X) > buffer_size:
                        safe_X = safe_X[-buffer_size:]
                        safe_y = safe_y[-buffer_size:]

                    # ✅ FIX: Dynamic batch composition based on available samples
                    while len(seizure_X) > 0 and len(safe_X) > 0:
                        # Calculate ideal counts
                        n_seizure_ideal = int(batch_size * target_ratio)
                        n_safe_ideal = batch_size - n_seizure_ideal

                        # ✅ CRITICAL: Limit to available samples
                        n_seizure = min(n_seizure_ideal, len(seizure_X))
                        n_safe = min(n_safe_ideal, len(safe_X))

                        # Ensure batch is full-sized (if not enough seizure, take more safe)
                        if n_seizure + n_safe < batch_size:
                            if len(safe_X) >= (batch_size - n_seizure):
                                n_safe = batch_size - n_seizure
                            elif len(seizure_X) >= (batch_size - n_safe):
                                n_seizure = batch_size - n_safe
                            else:
                                # Not enough data for a full batch, break and load more
                                break

                        # ✅ Sample with available data
                        if n_seizure > 0 and n_safe > 0:
                            seizure_idx = np.random.choice(len(seizure_X), n_seizure, replace=False)
                            safe_idx = np.random.choice(len(safe_X), n_safe, replace=False)

                            batch_X = np.concatenate([
                                seizure_X[seizure_idx],
                                safe_X[safe_idx]
                            ], axis=0)
                            batch_y = np.concatenate([
                                seizure_y[seizure_idx],
                                safe_y[safe_idx]
                            ], axis=0)

                            # Shuffle within batch
                            idx = np.arange(len(batch_X))
                            np.random.shuffle(idx)

                            yield batch_X[idx], batch_y[idx]

                            # Remove used samples
                            seizure_X = np.delete(seizure_X, seizure_idx, axis=0)
                            seizure_y = np.delete(seizure_y, seizure_idx, axis=0)
                            safe_X = np.delete(safe_X, safe_idx, axis=0)
                            safe_y = np.delete(safe_y, safe_idx, axis=0)
                        else:
                            break

                    # Update buffers
                    buffer_seizure_X = [seizure_X] if len(seizure_X) > 0 else []
                    buffer_seizure_y = [seizure_y] if len(seizure_y) > 0 else []
                    buffer_safe_X = [safe_X] if len(safe_X) > 0 else []
                    buffer_safe_y = [safe_y] if len(safe_y) > 0 else []

            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
                continue


def validation_generator(npz_paths, batch_size=32):
    """
    ✅ FIXED: Added infinite loop
    """
    while True:  # ← MUST BE HERE
        for path in npz_paths:
            try:
                X, y = load_npz_file(path)

                if len(X) == 0:
                    continue

                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i + batch_size]
                    batch_y = y[i:i + batch_size]

                    if len(batch_X) > 0:
                        yield batch_X, batch_y

            except Exception as e:
                print(f"⚠️ Validation error: {e}")
                continue


def count_samples(npz_paths):
    total = 0
    seizure_count = 0

    for p in npz_paths:
        try:
            data = np.load(p)
            y = data['y']
            total += len(y)
            seizure_count += np.sum(y == 1)
        except Exception as e:
            print(f"⚠️ Error counting {p}: {e}")
            continue

    return total, seizure_count


def focal_loss_fixed(alpha=0.70, gamma=2.5):
    """
    ✅ IMPROVED: Adjusted parameters
    alpha=0.70 (less aggressive), gamma=2.5 (more focus on hard examples)
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

        focal = alpha_t * focal_weight * bce
        return tf.reduce_mean(focal)

    return loss


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return 2 * (precision * recall) / (precision + recall + 1e-7)


def specificity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)

    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)

    return tn / (tn + fp + 1e-7)




def build_improved_eeg_cnn(input_shape):
    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv1D(64, 7, padding='same', kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.MaxPooling1D(2)(x)

    # Block 2
    x = layers.Conv1D(128, 5, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.MaxPooling1D(2)(x)

    # Block 3
    x = layers.Conv1D(256, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout1D(0.3)(x)

    # Attention
    attention = layers.Conv1D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])

    # Pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    # Dense
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inp, out)


model = build_improved_eeg_cnn((256, 18))


def plot_training_history(history):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)

    # Loss
    axes[0, 0].plot(epochs, hist['loss'], label='Train')
    axes[0, 0].plot(epochs, hist['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(epochs, hist['accuracy'], label='Train')
    axes[0, 1].plot(epochs, hist['val_accuracy'], label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # AUC
    axes[0, 2].plot(epochs, hist['auc'], label='Train')
    axes[0, 2].plot(epochs, hist['val_auc'], label='Val')
    axes[0, 2].set_title('AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Precision
    axes[1, 0].plot(epochs, hist['precision'], label='Train')
    axes[1, 0].plot(epochs, hist['val_precision'], label='Val')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Recall
    axes[1, 1].plot(epochs, hist['recall'], label='Train')
    axes[1, 1].plot(epochs, hist['val_recall'], label='Val')
    axes[1, 1].set_title('Recall (Sensitivity)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # F1
    axes[1, 2].plot(epochs, hist['f1_score'], label='Train')
    axes[1, 2].plot(epochs, hist['val_f1_score'], label='Val')
    axes[1, 2].set_title('F1 Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load paths
    train_paths = collect_npz_paths("dataset_final_gemini_v2/train")
    val_paths = collect_npz_paths("dataset_final_gemini_v2/val")

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Train files: {len(train_paths)}")
    print(f"Val files: {len(val_paths)}")

    if len(train_paths) == 0 or len(val_paths) == 0:
        print("\n❌ ERROR: No data files found!")
        exit(1)

    total_train, seizure_train = count_samples(train_paths)
    total_val, seizure_val = count_samples(val_paths)

    if total_train == 0 or total_val == 0:
        print("\n❌ ERROR: No samples loaded!")
        exit(1)

    print(f"\nTrain samples: {total_train:,}")
    print(f"  - Seizure: {seizure_train:,} ({100 * seizure_train / total_train:.2f}%)")
    print(f"  - Non-seizure: {total_train - seizure_train:,}")

    print(f"\nVal samples: {total_val:,}")
    print(f"  - Seizure: {seizure_val:,} ({100 * seizure_val / total_val:.2f}%)")
    print(f"  - Non-seizure: {total_val - seizure_val:,}")
    print("=" * 60)

    BATCH_SIZE = 32
    BUFFER_SIZE = 4096
    TARGET_RATIO = 0.15

    train_gen = balanced_batch_generator(train_paths, BATCH_SIZE, BUFFER_SIZE, TARGET_RATIO)
    val_gen = validation_generator(val_paths, BATCH_SIZE)

    steps_per_epoch = total_train // BATCH_SIZE
    val_steps = total_val // BATCH_SIZE

    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {val_steps}\n")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_recall',  # Changed from val_auc
        patience=15,  # Increased from 10
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        "best_eeg_seizure_model.keras",
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    # Build model
    model = build_improved_eeg_cnn((256, 18))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Lower LR
        loss=focal_loss_fixed(alpha=0.70, gamma=2.5),  # Adjusted alpha
        metrics=[
            'accuracy',
            metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            specificity,
            f1_score
        ]
    )
    model.summary()

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=50,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Plot results
    plot_training_history(history)

    # Save final model
    model.save("final_eeg_seizure_model.keras")
    print("\n✅ Models saved successfully")