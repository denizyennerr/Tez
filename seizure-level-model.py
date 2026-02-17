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


def natural_batch_generator(npz_paths, batch_size=32, shuffle_data=True):
    """
    ✅ NEW: Preserves natural class distribution in batches
    No artificial balancing - matches validation distribution
    """
    while True:
        paths = npz_paths.copy()
        if shuffle_data:
            random.shuffle(paths)

        for path in paths:
            try:
                X, y = load_npz_file(path)

                if len(X) == 0:
                    continue

                # Shuffle within file
                if shuffle_data:
                    indices = np.arange(len(X))
                    np.random.shuffle(indices)
                    X = X[indices]
                    y = y[indices]

                # Yield sequential batches
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i + batch_size]
                    batch_y = y[i:i + batch_size]

                    if len(batch_X) > 0:
                        yield batch_X, batch_y

            except Exception as e:
                print(f"⚠️ Error: {e}")
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


def weighted_binary_crossentropy(weight_for_1):
    """
    Custom weighted BCE loss that works with generators

    Args:
        weight_for_1: Weight multiplier for positive class (seizures)
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Standard BCE
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

        # Apply weights
        weights = y_true * weight_for_1 + (1 - y_true) * 1.0
        weighted_bce = bce * weights

        return tf.reduce_mean(weighted_bce)

    return loss


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


# ...existing imports...

if __name__ == '__main__':
    # Load paths
    train_paths = collect_npz_paths("dataset_final_gemini_v3/train")
    val_paths = collect_npz_paths("dataset_final_gemini_v3/val")

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

    # ✅ FIX: Calculate class weights based on VALIDATION distribution (not training)
    val_seizure_ratio = seizure_val / total_val
    weight_for_seizure = (1.0 - val_seizure_ratio) / val_seizure_ratio

    print(f"\n📊 Loss Weights:")
    print(f"  Non-seizure (0): 1.0")
    print(f"  Seizure (1):     {weight_for_seizure:.2f}")
    print(f"  Ratio (1:0):     {weight_for_seizure:.2f}:1\n")

    # ✅ FIX: Corrected generator calls
    train_gen = natural_batch_generator(train_paths, batch_size=BATCH_SIZE, shuffle_data=True)
    val_gen = validation_generator(val_paths, batch_size=BATCH_SIZE)

    steps_per_epoch = total_train // BATCH_SIZE
    val_steps = total_val // BATCH_SIZE

    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {val_steps}\n")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_f1_score',
        patience=20,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        "best_eeg_seizure_model.keras",
        monitor='val_f1_score',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # Build model
    model = build_improved_eeg_cnn((256, 18))

    # ✅ FIX: Compile with correct loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=weighted_binary_crossentropy(weight_for_seizure),  # ← Custom weighted loss
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

    # ✅ NEW: Full validation evaluation
    print("\n🔍 Evaluating on complete validation set...")

    # Collect all validation data
    X_val_list = []
    y_val_list = []

    for path in val_paths:
        try:
            X, y = load_npz_file(path)
            if len(X) > 0:
                X_val_list.append(X)
                y_val_list.append(y)
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            continue

    if X_val_list:
        X_val_full = np.concatenate(X_val_list, axis=0)
        y_val_full = np.concatenate(y_val_list, axis=0)

        # Predict
        print(f"\nPredicting on {len(y_val_full):,} validation samples...")
        y_pred_probs = model.predict(X_val_full, batch_size=64, verbose=1)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        # Metrics
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

        print("\n" + "=" * 60)
        print("FINAL VALIDATION REPORT")
        print("=" * 60)
        print(classification_report(y_val_full, y_pred,
                                    target_names=['Non-Seizure', 'Seizure'],
                                    digits=4))

        print(f"\nROC-AUC: {roc_auc_score(y_val_full, y_pred_probs):.4f}")

        cm = confusion_matrix(y_val_full, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"{'':15} {'Pred Non-Sz':>12} {'Pred Sz':>12}")
        print(f"{'True Non-Sz':<15} {cm[0, 0]:>12,} {cm[0, 1]:>12,}")
        print(f"{'True Sz':<15} {cm[1, 0]:>12,} {cm[1, 1]:>12,}")

        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\n📊 Key Metrics:")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity:          {specificity_val:.4f}")
        print(f"  True Positives:       {tp:,}")
        print(f"  False Negatives:      {fn:,} (missed seizures)")
        print(f"  False Positives:      {fp:,}")

    # Plot results
    plot_training_history(history)

    # Save final model
    model.save("final_eeg_seizure_model.keras")
    print("\n✅ Models saved successfully")