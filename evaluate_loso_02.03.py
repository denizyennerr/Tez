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
TIMESTAMP = "saved_outputs_play/20260318-152116_0.5s"

suffix = TIMESTAMP.replace('_', '-').split('-')[-1]

dataset_path = os.path.join(f'master_dataset_{suffix}.npz')
output_dir = TIMESTAMP
models_dir = os.path.join(output_dir, "models")
histories_dir = os.path.join(output_dir, "histories")
plots_dir = os.path.join(output_dir, "plots")
thresholds_dir = os.path.join(output_dir, "thresholds")  # Where val-set thresholds are stored

os.makedirs(plots_dir, exist_ok=True)

# Target sensitivity for operating-point threshold (tuned on val set)
TARGET_SENSITIVITY = 0.80


# =============================================================================
# %% Helper Functions
# =============================================================================

def find_optimal_threshold_from_val(y_val_true, y_val_prob, target_sensitivity=0.80):
    """
    Determine the optimal decision threshold using VALIDATION data only.

    Strategy: Among all thresholds that achieve >= target_sensitivity on the
    validation set, select the one that maximises precision (i.e. best F-beta
    operating point). Falls back to 0.5 if the target is unreachable.

    Parameters
    ----------
    y_val_true        : 1-D array of ground-truth binary labels (val set)
    y_val_prob        : 1-D array of predicted probabilities     (val set)
    target_sensitivity: float, minimum acceptable recall/sensitivity

    Returns
    -------
    float : chosen threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_val_true, y_val_prob)

    # precision_recall_curve appends a final point with no threshold;
    # align arrays so each index corresponds to a real threshold value.
    precision = precision[:-1]
    recall    = recall[:-1]

    valid_mask = recall >= target_sensitivity

    if not valid_mask.any():
        print(
            f"  ⚠️  Target sensitivity {target_sensitivity:.0%} is unreachable on "
            f"the validation set. Falling back to threshold = 0.5."
        )
        return 0.5

    # Among all thresholds satisfying the sensitivity constraint,
    # pick the one with the highest precision.
    best_idx = np.argmax(precision[valid_mask])
    best_threshold = thresholds[valid_mask][best_idx]
    return float(best_threshold)


def load_val_threshold(subject, thresholds_dir):
    """
    Load the pre-computed validation-set threshold for a subject.

    The threshold file is expected to be saved during the training phase
    (see save_val_threshold). If it does not exist, returns None.
    """
    path = os.path.join(thresholds_dir, f"threshold_subject_{subject}.npy")
    if os.path.exists(path):
        return float(np.load(path))
    return None


def save_val_threshold(subject, threshold, thresholds_dir):
    """Persist the val-set threshold so it can be loaded in evaluation."""
    os.makedirs(thresholds_dir, exist_ok=True)
    np.save(os.path.join(thresholds_dir, f"threshold_subject_{subject}.npy"), threshold)


# =============================================================================
# NOTE ─ If you are running training and evaluation in the SAME script you can
#         compute and cache the threshold right after validation inference:
#
#   y_val_prob = model.predict(X_val).ravel()
#   thresh = find_optimal_threshold_from_val(y_val_prob, y_val,
#                                            target_sensitivity=TARGET_SENSITIVITY)
#   save_val_threshold(subject, thresh, thresholds_dir)
#
#         Then in the evaluation loop below it is loaded without touching y_test.
# =============================================================================


# =============================================================================
# %% Plotting Functions
# =============================================================================

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


def plot_testing_evaluation(y_test, y_pred_prob, y_pred_class, threshold, subject, save_path):
    """
    Four-panel evaluation figure:
      1. Confusion Matrix          (operating-point at val-set threshold)
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
        f"Threshold = {threshold:.3f} (val-set, sens ≥ {TARGET_SENSITIVITY:.0%})",
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
    ax1.set_title("Confusion Matrix\n(secondary — operating point)")
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
    ax4.axvline(threshold, color="black", linestyle="--", lw=1.5,
                label=f"Threshold = {threshold:.3f}")
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
    ax2.plot(subjects, summary_df["sensitivity"],      "o-", color="tomato",    lw=2, label="Sensitivity")
    ax2.plot(subjects, summary_df["specificity"],      "s-", color="steelblue", lw=2, label="Specificity")
    ax2.plot(subjects, summary_df["balanced_accuracy"],"^-", color="seagreen",  lw=2, label="Balanced Accuracy")
    ax2.plot(subjects, summary_df["seizure_f1"],       "D-", color="orange",    lw=2, label="Seizure F1")
    ax2.axhline(TARGET_SENSITIVITY, color="gray", linestyle=":", lw=1.5,
                label=f"Target sensitivity = {TARGET_SENSITIVITY:.0%}")
    ax2.set_xticks(range(len(subjects)))
    ax2.set_xticklabels(subjects, rotation=45, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title(f"Secondary Metrics (Val-Set Threshold, Sensitivity ≥ {TARGET_SENSITIVITY:.0%})")
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

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_test = X[test_idx]
    y_test = y[test_idx]
    current_test_subject = groups[test_idx][0]

    model_path   = os.path.join(models_dir,   f"best_model_subject_{current_test_subject}.keras")
    history_path = os.path.join(histories_dir, f"history_subject_{current_test_subject}.csv")

    if not os.path.exists(model_path):
        print(f"⚠️  Skipping Subject {current_test_subject} — model not found.")
        continue

    print(f"\n{'='*60}")
    print(f"🧪  Evaluating Subject: {current_test_subject}")
    print(f"{'='*60}")

    # ── Load model and history ────────────────────────────────────────────────
    model      = tf.keras.models.load_model(model_path)
    history_df = pd.read_csv(history_path)

    # ── Test-set predictions (probabilities only — no threshold yet) ──────────
    y_pred_prob = model.predict(X_test, verbose=0).ravel()

    # =========================================================================
    # OPTION C — PRIMARY METRICS (threshold-independent)
    # =========================================================================
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    print(f"  [PRIMARY]   AUROC = {auroc:.4f}  |  AUPRC = {auprc:.4f}")

    # # =========================================================================
    # # OPTION B — SECONDARY METRICS (val-set threshold, never touching y_test)
    # # =========================================================================
    #
    # --- Attempt 1: load a pre-saved threshold from the training phase --------
    threshold = load_val_threshold(current_test_subject, thresholds_dir)
    #
    # --- Attempt 2: if not pre-saved, recompute from val split of TRAIN data --
    # (uses only training indices, so y_test remains untouched)
    if threshold is None:
        print(
            f"  ℹ️  No pre-saved threshold found for subject {current_test_subject}. "
            f"Computing from a held-out validation split of the training fold..."
        )
        X_train_full = X[train_idx]
        y_train_full = y[train_idx]

        # 80/20 split of the training fold for threshold tuning
        n_train_samples = len(y_train_full)
        split_point     = int(n_train_samples * 0.80)
        # Preserve temporal order — do NOT shuffle EEG epochs
        X_tr, X_val = X_train_full[:split_point], X_train_full[split_point:]
        y_tr, y_val = y_train_full[:split_point], y_train_full[split_point:]

        y_val_prob = model.predict(X_val, verbose=0).ravel()
        threshold  = find_optimal_threshold_from_val(
            y_val, y_val_prob, target_sensitivity=TARGET_SENSITIVITY
        )
        # Cache it for reproducibility
        save_val_threshold(current_test_subject, threshold, thresholds_dir)

    print(f"  [SECONDARY] Threshold = {threshold:.4f}  (val-set, sens ≥ {TARGET_SENSITIVITY:.0%})")

    # Apply threshold to test predictions
    y_pred_class = (y_pred_prob > threshold).astype(int)

    # ── Classification metrics at the chosen operating point ─────────────────
    report_dict  = classification_report(
        y_test, y_pred_class,
        target_names=["Normal", "Seizure"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    print("\n  Classification Report (operating-point):")
    display(report_df)

    cm = confusion_matrix(y_test, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    sensitivity    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    specificity    = tn / (tn + fp)   if (tn + fp) > 0 else 0.0
    balanced_acc   = balanced_accuracy_score(y_test, y_pred_class)
    seizure_f1     = report_dict["Seizure"]["f1-score"]

    print(f"  Sensitivity    = {sensitivity:.4f}")
    print(f"  Specificity    = {specificity:.4f}")
    print(f"  Balanced Acc   = {balanced_acc:.4f}")
    print(f"  Seizure F1     = {seizure_f1:.4f}")

    all_reports.append({
        "subject":          current_test_subject,
        "threshold":        threshold,
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
        y_test, y_pred_prob, y_pred_class, threshold, current_test_subject,
        os.path.join(plots_dir, f"subject_{current_test_subject}_testing_evaluation.png"),
    )

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