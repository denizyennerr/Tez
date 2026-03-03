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
import tensorflow.keras.backend as K # Added for memory management

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
# %% Updated base paths
dataset_path = 'master_dataset_2s.npz'

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join("saved_outputs", timestamp)

models_dir = os.path.join(output_dir, "models")
histories_dir = os.path.join(output_dir, "histories")
log_dir = os.path.join(output_dir, "logs")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(histories_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ==========================================

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 100
DECISION_THRESHOLD = 0.3

# %% High-Performance Data Pipeline
def create_tf_dataset(X, y, batch_size=64, is_training=False):
    """
    Creates an optimized tf.data pipeline to prevent GPU starvation.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))

    if is_training:
        dataset = dataset.shuffle(buffer_size=1024)

    dataset = dataset.batch(batch_size)

    # SENIOR REVIEW NOTE: Removed .cache() here.
    # Caching large datasets in RAM across a 24-fold loop is what crashes Colab's System RAM.
    # Prefetch alone is enough to keep the GPU busy without blowing up memory.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

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
            tf.keras.metrics.AUC(name='auc'), # AUC is threshold-agnostic
            tf.keras.metrics.Precision(name='precision', thresholds=threshold),
            tf.keras.metrics.Recall(name='recall', thresholds=threshold)
        ]
    )
    return model

# %% Data Loading
print("Loading data...")

data = np.load(dataset_path)
X, y, groups = data['X'], data['y'], data['s']

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
    print(f"\n🚀 Training holding out Subject: {current_test_subject}")

    # Subject-level Validation Split
    unique_train_subjects = np.unique(groups_train)
    val_subject = unique_train_subjects[0]
    val_mask = (groups_train == val_subject)
    train_mask = ~val_mask

    X_train, X_val = X_train_full[train_mask], X_train_full[val_mask]
    y_train, y_val = y_train_full[train_mask], y_train_full[val_mask]

    # Create optimized datasets
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=BATCH_SIZE, is_training=True)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=BATCH_SIZE, is_training=False)

    # Build model with fixed learning rate
    model = build_seizure_model_cnn(input_shape, learning_rate=LEARNING_RATE,
    threshold= DECISION_THRESHOLD)

    fold_log_dir = os.path.join(log_dir, f"subject_{current_test_subject}")
    model_save_path = os.path.join(models_dir, f"best_model_subject_{current_test_subject}.keras")

    # EarlyStopping patience biraz yüksek tutuldu (25) ki model zor öğreniyorsa bile şansı olsun.
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=fold_log_dir, histogram_freq=1),
        ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]

    # Train using the tf.data pipeline
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(histories_dir, f"history_subject_{current_test_subject}.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"✅ Saved history for subject {current_test_subject}")

    # ==========================================
    # 🧹 SENIOR DS MEMORY CLEANUP
    # ==========================================
    # 1. Delete large variables created in this iteration
    del X_train_full, X_test, y_train_full, y_test
    del X_train, X_val, y_train, y_val
    del train_dataset, val_dataset
    del model, history

    # 2. Clear the Keras backend (destroys the computational graph from RAM)
    K.clear_session()

    # 3. Force Python to run garbage collection immediately
    gc.collect()

print("\n🎉 All training complete. Models and histories saved successfully.")

# ==========================================



# %% Imports and Configuration
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from IPython.display import display

# --- UPDATE THESE TO MATCH YOUR TRAINING RUN ---
TIMESTAMP = "/kaggle/working/Tez/saved_outputs/20260303-083816" # Updated to your recent run
dataset_path = '/kaggle/input/datasets/denizyennerr99/4second-epoch-masterdataset/master_dataset_4s.npz'
output_dir = os.path.join("Tez/saved_outputs", TIMESTAMP)

models_dir = os.path.join(output_dir, "models")
histories_dir = os.path.join(output_dir, "histories")
plots_dir = os.path.join(output_dir, "plots")

os.makedirs(plots_dir, exist_ok=True)

# =============================================================================
# 🚀 SENIOR DS CONFIGURATION (Synced with Training)
# =============================================================================
BATCH_SIZE = 128
DECISION_THRESHOLD = 0.3

# =============================================================================
# %% Plotting Functions
# =============================================================================
def calculate_average_history(all_histories):
    """
    Averages training histories across all subjects.
    Handles variable epoch lengths (due to EarlyStopping) by forward-filling
    the last recorded metrics for subjects that stopped early.
    """
    if not all_histories:
        return None

    # Find the maximum number of epochs any subject trained for
    max_epochs = max(len(df) for df in all_histories)

    aligned_histories = []
    for df in all_histories:
        # Reindex to max_epochs, which introduces NaNs for early-stopped runs
        aligned_df = df.reindex(range(max_epochs))
        # Forward fill the NaNs with the last valid epoch's metrics
        aligned_df = aligned_df.ffill()
        aligned_histories.append(aligned_df)

    # Concatenate and group by the index (epoch) to find the mean
    concat_df = pd.concat(aligned_histories)
    avg_history_df = concat_df.groupby(concat_df.index).mean()

    return avg_history_df


def plot_average_training_history(avg_history_df, save_path):
    """Plots the averaged training and validation curves across all folds."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Average Training History Across All Subjects", fontsize=16, fontweight="bold")

    # Plot Accuracy
    axs[0].plot(avg_history_df["accuracy"],     label="Train Accuracy",      color="#1f77b4", lw=2)
    axs[0].plot(avg_history_df["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", lw=2)
    axs[0].set_title("Average Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Plot Loss
    axs[1].plot(avg_history_df["loss"],     label="Train Loss",      color="#1f77b4", lw=2)
    axs[1].plot(avg_history_df["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
    axs[1].set_title("Average Model Loss (Binary Cross-Entropy)")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_training_history(history_df, subject, save_path):
    """Training / validation accuracy and loss curves."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History | Subject: {subject}", fontsize=16, fontweight="bold")

    axs[0].plot(history_df["accuracy"],     label="Train",      color="#1f77b4", lw=2)
    axs[0].plot(history_df["val_accuracy"], label="Validation", color="#ff7f0e", lw=2)
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    axs[1].plot(history_df["loss"],     label="Train",      color="#1f77b4", lw=2)
    axs[1].plot(history_df["val_loss"], label="Validation", color="#ff7f0e", lw=2)
    axs[1].set_title("Model Loss (Binary Cross-Entropy)")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_testing_evaluation(y_test, y_pred_prob, y_pred_class, subject, save_path):
    """
    Four-panel evaluation figure:
      1. Confusion Matrix          (operating-point at 0.3 threshold)
      2. ROC Curve + AUROC         (threshold-independent, PRIMARY metric)
      3. Precision-Recall + AUPRC  (threshold-independent, PRIMARY metric)
      4. Probability Distribution  (visual sanity check)
    """
    auroc      = roc_auc_score(y_test, y_pred_prob)
    auprc      = average_precision_score(y_test, y_pred_prob)
    fpr, tpr, _= roc_curve(y_test, y_pred_prob)
    prec, rec, _= precision_recall_curve(y_test, y_pred_prob)
    cm         = confusion_matrix(y_test, y_pred_class)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Testing Evaluation | Subject: {subject} | "
        f"Fixed Threshold = {DECISION_THRESHOLD}",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Confusion Matrix ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
        xticklabels=["Normal", "Seizure"],
        yticklabels=["Normal", "Seizure"],
        annot_kws={"size": 14},
    )
    ax1.set_title(f"Confusion Matrix\n(Threshold = {DECISION_THRESHOLD})")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    # ── Panel 2: ROC Curve ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUROC = {auroc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax2.set_xlabel("False Positive Rate (1 − Specificity)")
    ax2.set_ylabel("True Positive Rate (Sensitivity)")
    ax2.set_title("ROC Curve\n(PRIMARY metric)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # ── Panel 3: Precision-Recall Curve ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rec, prec, color="purple", lw=2, label=f"AUPRC = {auprc:.4f}")
    baseline = y_test.mean()          # prevalence = no-skill baseline
    ax3.axhline(baseline, color="gray", linestyle="--", lw=1,
                label=f"No-skill baseline = {baseline:.3f}")
    ax3.set_xlabel("Recall (Sensitivity)")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve\n(PRIMARY metric — better for imbalanced data)")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # ── Panel 4: Predicted Probability Distribution ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(y_pred_prob[y_test == 0], bins=50, alpha=0.6,
             color="steelblue", label="Normal", density=True)
    ax4.hist(y_pred_prob[y_test == 1], bins=50, alpha=0.6,
             color="tomato",    label="Seizure", density=True)
    ax4.axvline(DECISION_THRESHOLD, color="black", linestyle="--", lw=1.5,
                label=f"Threshold = {DECISION_THRESHOLD}")
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.set_title("Predicted Probability Distribution")
    ax4.legend()
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_aggregate_summary(summary_df, save_path):
    """
    Bar chart of per-subject primary metrics (AUROC, AUPRC) with mean line,
    plus secondary operating-point metrics.
    """
    subjects = summary_df["subject"].astype(str)
    x = np.arange(len(subjects))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(subjects) * 0.9), 12))
    fig.suptitle("LOSO Aggregate Performance — CHB-MIT Seizure Detection",
                 fontsize=15, fontweight="bold")

    # ── Top: Primary (threshold-independent) metrics ──────────────────────────
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, summary_df["auroc"], width,
                   label="AUROC", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, summary_df["auprc"], width,
                   label="AUPRC", color="purple",    alpha=0.85)
    ax.axhline(summary_df["auroc"].mean(), color="steelblue", linestyle="--",
               lw=1.5, label=f"Mean AUROC = {summary_df['auroc'].mean():.4f}")
    ax.axhline(summary_df["auprc"].mean(), color="purple",    linestyle="--",
               lw=1.5, label=f"Mean AUPRC = {summary_df['auprc'].mean():.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Primary Metrics (Threshold-Independent)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # ── Bottom: Secondary (operating-point) metrics ───────────────────────────
    ax2 = axes[1]
    ax2.plot(subjects, summary_df["sensitivity"],      "o-", color="tomato",    lw=2, label="Sensitivity (Recall)")
    ax2.plot(subjects, summary_df["specificity"],      "s-", color="steelblue", lw=2, label="Specificity")
    ax2.plot(subjects, summary_df["balanced_accuracy"],"^-", color="seagreen",  lw=2, label="Balanced Accuracy")
    ax2.plot(subjects, summary_df["seizure_f1"],       "D-", color="orange",    lw=2, label="Seizure F1")
    ax2.set_xticks(range(len(subjects)))
    ax2.set_xticklabels(subjects, rotation=45, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title(f"Secondary Metrics (Fixed Threshold = {DECISION_THRESHOLD})")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# %% Data Loading
# =============================================================================
data   = np.load(dataset_path)
X, y, groups = data["X"], data["y"], data["s"]
logo   = LeaveOneGroupOut()

# =============================================================================
# %% LOSO Evaluation Loop
# =============================================================================
all_reports = []
all_histories = []

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_test = X[test_idx]
    y_test = y[test_idx]
    current_test_subject = groups[test_idx][0]

    model_path   = os.path.join(models_dir,   f"best_model_subject_{current_test_subject}.keras")
    history_path = os.path.join(histories_dir, f"history_subject_{current_test_subject}.csv")
    all_histories.append(history_df)

    if not os.path.exists(model_path):
        print(f"⚠️  Skipping Subject {current_test_subject} — model not found.")
        continue

    print(f"\n{'='*60}")
    print(f"🧪  Evaluating Subject: {current_test_subject}")
    print(f"{'='*60}")

    # ── Load model and history ────────────────────────────────────────────────
    model      = tf.keras.models.load_model(model_path)
    history_df = pd.read_csv(history_path)

    # ── Test-set predictions ──────────────────────────────────────────────────
    # Note: Utilizing the updated BATCH_SIZE for faster inference
    y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()

    # Apply the new fixed threshold explicitly
    y_pred_class = (y_pred_prob >= DECISION_THRESHOLD).astype(int)

    # =========================================================================
    # PRIMARY METRICS (threshold-independent)
    # =========================================================================
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    print(f"  [PRIMARY]   AUROC = {auroc:.4f}  |  AUPRC = {auprc:.4f}")

    # =========================================================================
    # SECONDARY METRICS (operating-point)
    # =========================================================================
    print(f"  [SECONDARY] Fixed Threshold = {DECISION_THRESHOLD}")

    report_dict  = classification_report(
        y_test, y_pred_class,
        target_names=["Normal", "Seizure"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    print("\n  Classification Report:")
    display(report_df)

    cm = confusion_matrix(y_test, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    sensitivity    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    specificity    = tn / (tn + fp)   if (tn + fp) > 0 else 0.0
    balanced_acc   = balanced_accuracy_score(y_test, y_pred_class)
    seizure_f1     = report_dict["Seizure"]["f1-score"]

    print(f"  Sensitivity (Recall) = {sensitivity:.4f}")
    print(f"  Specificity          = {specificity:.4f}")
    print(f"  Balanced Acc         = {balanced_acc:.4f}")
    print(f"  Seizure F1           = {seizure_f1:.4f}")

    all_reports.append({
        "subject":          current_test_subject,
        "threshold":        DECISION_THRESHOLD,
        # PRIMARY (threshold-independent)
        "auroc":            auroc,
        "auprc":            auprc,
        # SECONDARY (operating-point)
        "accuracy":         report_dict["accuracy"],
        "balanced_accuracy":balanced_acc,
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "seizure_f1":       seizure_f1,
    })

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_training_history(
        history_df, current_test_subject,
        os.path.join(plots_dir, f"subject_{current_test_subject}_training_history.png"),
    )
    plot_testing_evaluation(
        y_test, y_pred_prob, y_pred_class, current_test_subject,
        os.path.join(plots_dir, f"subject_{current_test_subject}_testing_evaluation.png"),
    )

    # Free memory
    del model
    tf.keras.backend.clear_session()

# =============================================================================
# %% Aggregate Results Summary
# =============================================================================
if all_reports:
    summary_df = pd.DataFrame(all_reports)

    print(f"\n{'='*60}")
    print("🏆  Overall LOSO Summary")
    print(f"{'='*60}")
    display(summary_df)

    summary_csv_path = os.path.join(output_dir, "overall_loso_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n✅  Summary saved to: {summary_csv_path}")

    # ── Aggregate statistics ──────────────────────────────────────────────────
    print("\n📊  Aggregate Statistics (mean ± std across subjects):")
    metrics = ["auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "seizure_f1"]
    col_width = 22
    print(f"  {'Metric':<{col_width}} {'Mean':>8}   {'Std':>8}   {'Min':>8}   {'Max':>8}")
    print(f"  {'-'*60}")
    for m in metrics:
        print(
            f"  {m:<{col_width}} "
            f"{summary_df[m].mean():>8.4f}   "
            f"{summary_df[m].std():>8.4f}   "
            f"{summary_df[m].min():>8.4f}   "
            f"{summary_df[m].max():>8.4f}"
        )

    # ── Aggregate plot ────────────────────────────────────────────────────────
    plot_aggregate_summary(
        summary_df,
        os.path.join(plots_dir, "aggregate_loso_summary.png"),
    )
    print(f"\n✅  Aggregate plot saved to: {os.path.join(plots_dir, 'aggregate_loso_summary.png')}")

    # ── NEW: Average Training History Plot ────────────────────────────────────
    avg_history_df = calculate_average_history(all_histories)
    if avg_history_df is not None:
        avg_history_path = os.path.join(plots_dir, "aggregate_training_history.png")
        plot_average_training_history(avg_history_df, avg_history_path)
        print(f"✅  Average training history plot saved to: {avg_history_path}")