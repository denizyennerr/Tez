# %% Imports and Configuration
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

TIMESTAMP = "20260309-155522_10s"

suffix = TIMESTAMP.replace('_', '-').split('-')[-1]

dataset_path = os.path.join(f'master_dataset_{suffix}.npz')
output_dir = os.path.join("saved_outputs", TIMESTAMP)

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
    axs[0].plot(avg_history_df["accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
    axs[0].plot(avg_history_df["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", lw=2)
    axs[0].set_title("Average Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Plot Loss
    axs[1].plot(avg_history_df["loss"], label="Train Loss", color="#1f77b4", lw=2)
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

    axs[0].plot(history_df["accuracy"], label="Train", color="#1f77b4", lw=2)
    axs[0].plot(history_df["val_accuracy"], label="Validation", color="#ff7f0e", lw=2)
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    axs[1].plot(history_df["loss"], label="Train", color="#1f77b4", lw=2)
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
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred_class)

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
             color="tomato", label="Seizure", density=True)
    ax4.axvline(DECISION_THRESHOLD, color="black", linestyle="--", lw=1.5,
                label=f"Threshold = {DECISION_THRESHOLD}")
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.set_title("Predicted Probability Distribution")
    ax4.legend()
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


import matplotlib.pyplot as plt
import numpy as np


def plot_primary_metrics(summary_df, save_path):
    """
    Bar chart of per-subject primary metrics (AUROC, AUPRC) with mean lines.
    Saved as an independent standalone figure.
    """
    subjects = summary_df["subject"].astype(str)
    x = np.arange(len(subjects))
    width = 0.35

    # Adjusted figsize to be optimal for a single wide plot
    fig, ax = plt.subplots(figsize=(max(12, len(subjects) * 0.6), 6))
    fig.suptitle("LOSO Performance Metrics of CHB-MIT Seizure Detection Model",
                 fontsize=15, fontweight="bold")

    ax.bar(x - width / 2, summary_df["auroc"], width,
           label="AUROC", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, summary_df["auprc"], width,
           label="AUPRC", color="purple", alpha=0.85)

    ax.axhline(summary_df["auroc"].mean(), color="steelblue", linestyle="--",
               lw=1.5, label=f"Mean AUROC = {summary_df['auroc'].mean():.4f}")
    ax.axhline(summary_df["auprc"].mean(), color="purple", linestyle="--",
               lw=1.5, label=f"Mean AUPRC = {summary_df['auprc'].mean():.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Primary Metrics")
    ax.legend(loc="lower left", fontsize=9)

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_secondary_metrics(summary_df, save_path, DECISION_THRESHOLD=0.3):
    """
    Grouped bar chart of secondary operating-point metrics.
    Saved as an independent standalone figure.
    """
    subjects = summary_df["subject"].astype(str)
    x = np.arange(len(subjects))
    width = 0.20

    # Needs a slightly wider base width to accommodate 4 bars per subject
    fig, ax = plt.subplots(figsize=(max(14, len(subjects) * 1.0), 6))
    fig.suptitle("LOSO Performance Metrics of CHB-MIT Seizure Detection Model",
                 fontsize=15, fontweight="bold")

    offset1 = -1.5 * width
    offset2 = -0.5 * width
    offset3 = 0.5 * width
    offset4 = 1.5 * width

    ax.bar(x + offset1, summary_df["sensitivity"], width,
           label="Sensitivity (Recall)", color="tomato", alpha=0.85)
    ax.bar(x + offset2, summary_df["specificity"], width,
           label="Specificity", color="steelblue", alpha=0.85)
    ax.bar(x + offset3, summary_df["balanced_accuracy"], width,
           label="Balanced Accuracy", color="seagreen", alpha=0.85)
    ax.bar(x + offset4, summary_df["seizure_f1"], width,
           label="Seizure F1", color="orange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Secondary Performance Metrics")

    ax.legend(loc="lower left", fontsize=9, ncol=4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
# =============================================================================
# %% Data Loading
# =============================================================================
print(f"Loading dataset from: {dataset_path}")
data = np.load(dataset_path)
X, y, groups = data["X"], data["y"], data["s"]
logo = LeaveOneGroupOut()

# =============================================================================
# %% LOSO Evaluation Loop
# =============================================================================
all_reports = []
all_histories = []

print(f"📁 Searching for models in: {models_dir}")

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_test = X[test_idx]
    y_test = y[test_idx]
    current_test_subject = groups[test_idx][0]

    model_path = os.path.join(models_dir, f"best_model_subject_{current_test_subject}.keras")
    history_path = os.path.join(histories_dir, f"history_subject_{current_test_subject}.csv")

    if not os.path.exists(model_path):
        print(f"⚠️  Skipping Subject {current_test_subject} — model not found at: {model_path}")
        continue

    print(f"\n{'=' * 60}")
    print(f"🧪  Evaluating Subject: {current_test_subject}")
    print(f"{'=' * 60}")

    # ── Load model and history ────────────────────────────────────────────────
    model = tf.keras.models.load_model(model_path)
    history_df = pd.read_csv(history_path)

    # 2. FIXED: Append AFTER loading the data, not before!
    all_histories.append(history_df)

    # ── Test-set predictions ──────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_pred_class = (y_pred_prob >= DECISION_THRESHOLD).astype(int)

    # =========================================================================
    # PRIMARY METRICS (threshold-independent)
    # =========================================================================
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    print(f" AUROC = {auroc:.4f}  |  AUPRC = {auprc:.4f}")

    # =========================================================================
    # SECONDARY METRICS (operating-point)
    # =========================================================================
    print(f"  Fixed Threshold = {DECISION_THRESHOLD}")

    report_dict = classification_report(
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

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_test, y_pred_class)
    seizure_f1 = report_dict["Seizure"]["f1-score"]

    print(f"  Sensitivity (Recall) = {sensitivity:.4f}")
    print(f"  Specificity          = {specificity:.4f}")
    print(f"  Balanced Acc         = {balanced_acc:.4f}")
    print(f"  Seizure F1           = {seizure_f1:.4f}")

    all_reports.append({
        "subject": current_test_subject,
        "threshold": DECISION_THRESHOLD,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": report_dict["accuracy"],
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "seizure_f1": seizure_f1,
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

    del model
    tf.keras.backend.clear_session()

# =============================================================================
# %% Aggregate Results Summary
# =============================================================================
if all_reports:
    summary_df = pd.DataFrame(all_reports)

    print(f"\n{'=' * 60}")
    print("🏆  Overall LOSO Summary")
    print(f"{'=' * 60}")
    display(summary_df)

    summary_csv_path = os.path.join(output_dir, f"overall_loso_summary_{suffix}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n✅  Summary saved to: {summary_csv_path}")

    print("\n📊  Aggregate Statistics (mean ± std across subjects):")
    metrics = ["auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "seizure_f1"]
    col_width = 22
    print(f"  {'Metric':<{col_width}} {'Mean':>8}   {'Std':>8}   {'Min':>8}   {'Max':>8}")
    print(f"  {'-' * 60}")
    for m in metrics:
        print(
            f"  {m:<{col_width}} "
            f"{summary_df[m].mean():>8.4f}   "
            f"{summary_df[m].std():>8.4f}   "
            f"{summary_df[m].min():>8.4f}   "
            f"{summary_df[m].max():>8.4f}"
        )

    primary_plot_path = os.path.join(plots_dir, f"aggregate_primary_metrics_{suffix}.png")
    secondary_plot_path = os.path.join(plots_dir, f"aggregate_secondary_metrics_{suffix}.png")

    plot_primary_metrics(summary_df, primary_plot_path)
    print(f"\n✅  Primary aggregate plot saved to: {primary_plot_path}")

    plot_secondary_metrics(summary_df, secondary_plot_path, DECISION_THRESHOLD)
    print(f"✅  Secondary aggregate plot saved to: {secondary_plot_path}")

    avg_history_df = calculate_average_history(all_histories)
    if avg_history_df is not None:
        avg_history_path = os.path.join(plots_dir, f"aggregate_training_history_{suffix}.png")
        plot_average_training_history(avg_history_df, avg_history_path)
        print(f"✅  Average training history plot saved to: {avg_history_path}")