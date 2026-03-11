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

TIMESTAMP = "20260303-162053-2s"
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
    if not all_histories:
        return None
    max_epochs = max(len(df) for df in all_histories)
    aligned_histories = []
    for df in all_histories:
        aligned_df = df.reindex(range(max_epochs)).ffill()
        aligned_histories.append(aligned_df)
    concat_df = pd.concat(aligned_histories)
    return concat_df.groupby(concat_df.index).mean()


def plot_average_training_history(avg_history_df, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Average Training History Across All Subjects", fontsize=16, fontweight="bold")
    axs[0].plot(avg_history_df["accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
    axs[0].plot(avg_history_df["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", lw=2)
    axs[0].set_title("Average Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, linestyle="--", alpha=0.6)
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
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred_class)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Testing Evaluation | Subject: {subject} ", fontsize=14,
                 fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1, xticklabels=["Normal", "Seizure"],
                yticklabels=["Normal", "Seizure"], annot_kws={"size": 14})
    ax1.set_title(f"Confusion Matrix\n(Threshold = {DECISION_THRESHOLD})")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUROC = {auroc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate ")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rec, prec, color="purple", lw=2, label=f"AUPRC = {auprc:.4f}")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.6)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(y_pred_prob[y_test == 0], bins=50, alpha=0.6, color="steelblue", label="Normal", density=True)
    ax4.hist(y_pred_prob[y_test == 1], bins=50, alpha=0.6, color="tomato", label="Seizure", density=True)
    ax4.axvline(DECISION_THRESHOLD, color="black", linestyle="--", lw=1.5, label=f"Threshold = {DECISION_THRESHOLD}")
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.set_title("Predicted Probability Distribution")
    ax4.legend()
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_aggregate_testing_evaluation(y_test_list, y_pred_prob_list, y_pred_class_list, save_path):
    """
    Plots the global aggregate evaluation. Modifed to show individual patient
    traces and shaded standard deviation areas for ROC and PR curves.
    """
    # 1. Prepare global arrays for Confusion Matrix and Histogram
    y_test_all = np.concatenate(y_test_list)
    y_pred_prob_all = np.concatenate(y_pred_prob_list)
    y_pred_class_all = np.concatenate(y_pred_class_list)

    cm = confusion_matrix(y_test_all, y_pred_class_all)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Aggregate Testing Evaluation | All Subjects", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Panel 1: Confusion Matrix ---
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1, xticklabels=["Normal", "Seizure"],
                yticklabels=["Normal", "Seizure"], annot_kws={"size": 16})
    ax1.set_title(f"Aggregate Confusion Matrix", fontsize=14)
    ax1.set_ylabel("True Label", fontsize=12)
    ax1.set_xlabel("Predicted Label", fontsize=12)

    # --- Panel 2: ROC Curve (with shaded std dev) ---
    ax2 = fig.add_subplot(gs[0, 1])
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for y_t, y_p in zip(y_test_list, y_pred_prob_list):
        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = roc_auc_score(y_t, y_p)
        aucs.append(roc_auc)
        # Interpolate TPR to a common FPR scale
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        # Plot individual faint line
        ax2.plot(fpr, tpr, color='steelblue', lw=1, alpha=0.2)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Plot Mean ROC
    ax2.plot(mean_fpr, mean_tpr, color='blue', lw=2.5, label=f"Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})")

    # Shade Standard Deviation Area
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax2.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3, label=r'$\pm$ 1 std. dev.')

    ax2.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random classifier")
    ax2.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
    ax2.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax2.set_title("Aggregate ROC Curve", fontsize=14)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # --- Panel 3: Precision-Recall Curve (with shaded std dev) ---
    ax3 = fig.add_subplot(gs[1, 0])
    precs = []
    pr_aucs = []
    mean_recall = np.linspace(0, 1, 100)

    for y_t, y_p in zip(y_test_list, y_pred_prob_list):
        prec, rec, _ = precision_recall_curve(y_t, y_p)
        pr_auc = average_precision_score(y_t, y_p)
        pr_aucs.append(pr_auc)

        # precision_recall_curve returns decreasing recall. Reverse them for np.interp
        rec, prec = rec[::-1], prec[::-1]
        interp_prec = np.interp(mean_recall, rec, prec)
        precs.append(interp_prec)
        # Plot individual faint line
        ax3.plot(rec, prec, color='purple', lw=1, alpha=0.2)

    mean_prec = np.mean(precs, axis=0)
    mean_pr_auc = np.mean(pr_aucs)
    std_pr_auc = np.std(pr_aucs)

    # Plot Mean PRC
    ax3.plot(mean_recall, mean_prec, color='indigo', lw=2.5,
             label=f"Mean PRC (AUC = {mean_pr_auc:.3f} $\pm$ {std_pr_auc:.3f})")

    # Shade Standard Deviation Area
    std_prec = np.std(precs, axis=0)
    precs_upper = np.minimum(mean_prec + std_prec, 1)
    precs_lower = np.maximum(mean_prec - std_prec, 0)
    ax3.fill_between(mean_recall, precs_lower, precs_upper, color='grey', alpha=0.3, label=r'$\pm$ 1 std. dev.')

    ax3.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax3.set_ylabel("Precision", fontsize=12)
    ax3.set_title("Aggregate Precision-Recall Curve", fontsize=14)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # --- Panel 4: Probability Distribution ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(y_pred_prob_all[y_test_all == 0], bins=100, alpha=0.6, color="steelblue", label="Normal", density=True)
    ax4.hist(y_pred_prob_all[y_test_all == 1], bins=100, alpha=0.6, color="tomato", label="Seizure", density=True)
    ax4.axvline(DECISION_THRESHOLD, color="black", linestyle="--", lw=2, label=f"Threshold = {DECISION_THRESHOLD}")
    ax4.set_xlabel("Predicted Probability", fontsize=12)
    ax4.set_ylabel("Density", fontsize=12)
    ax4.set_title("Aggregate Predicted Probability Distribution", fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_primary_metrics(summary_df, save_path):
    subjects = summary_df["subject"].astype(str)
    x = np.arange(len(subjects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(subjects) * 0.6), 6))
    fig.suptitle(f"LOSO Performance Metrics of {suffix} - CHB-MIT Seizure Detection Model", fontsize=15,
                 fontweight="bold")
    ax.bar(x - width / 2, summary_df["auroc"], width, label="AUROC", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, summary_df["auprc"], width, label="AUPRC", color="purple", alpha=0.85)
    ax.axhline(summary_df["auroc"].mean(), color="steelblue", linestyle="--", lw=1.5,
               label=f"Mean AUROC = {summary_df['auroc'].mean():.4f}")
    ax.axhline(summary_df["auprc"].mean(), color="purple", linestyle="--", lw=1.5,
               label=f"Mean AUPRC = {summary_df['auprc'].mean():.4f}")
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
    subjects = summary_df["subject"].astype(str)
    x = np.arange(len(subjects))
    width = 0.20
    fig, ax = plt.subplots(figsize=(max(14, len(subjects) * 1.0), 6))
    fig.suptitle(f"LOSO Performance Metrics of {suffix} - CHB-MIT Seizure Detection Model", fontsize=15,
                 fontweight="bold")
    offset1, offset2, offset3, offset4 = -1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width
    ax.bar(x + offset1, summary_df["sensitivity"], width, label="Sensitivity (Recall)", color="tomato", alpha=0.85)
    ax.bar(x + offset2, summary_df["specificity"], width, label="Specificity", color="steelblue", alpha=0.85)
    ax.bar(x + offset3, summary_df["balanced_accuracy"], width, label="Balanced Accuracy", color="seagreen", alpha=0.85)
    ax.bar(x + offset4, summary_df["seizure_f1"], width, label="Seizure F1", color="orange", alpha=0.85)
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

global_y_test_list = []
global_y_pred_prob_list = []
global_y_pred_class_list = []

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

    model = tf.keras.models.load_model(model_path)
    history_df = pd.read_csv(history_path)
    all_histories.append(history_df)

    # ── Test-set predictions ──────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_pred_class = (y_pred_prob >= DECISION_THRESHOLD).astype(int)

    # 🌟 NEW: Append the predictions to global lists AFTER they are generated
    global_y_test_list.append(y_test)
    global_y_pred_prob_list.append(y_pred_prob)
    global_y_pred_class_list.append(y_pred_class)

    # =========================================================================
    # PRIMARY METRICS (threshold-independent)
    # =========================================================================
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    print(f" AUROC = {auroc:.4f}  |  AUPRC = {auprc:.4f}")

    print(f"  Fixed Threshold = {DECISION_THRESHOLD}")
    report_dict = classification_report(y_test, y_pred_class, target_names=["Normal", "Seizure"], output_dict=True,
                                        zero_division=0)
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

    # ── Save individual plots ──────────────────────────────────────────────────
    plot_training_history(history_df, current_test_subject,
                          os.path.join(plots_dir, f"subject_{current_test_subject}_training_history.png"))
    plot_testing_evaluation(y_test, y_pred_prob, y_pred_class, current_test_subject,
                            os.path.join(plots_dir, f"subject_{current_test_subject}_testing_evaluation.png"))

    del model
    tf.keras.backend.clear_session()

# =============================================================================
# %% Aggregate Results Summary & Global Plots
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
            f"  {m:<{col_width}} {summary_df[m].mean():>8.4f}   {summary_df[m].std():>8.4f}   {summary_df[m].min():>8.4f}   {summary_df[m].max():>8.4f}")

    # 🌟 UPDATED: Generating Aggregate Evaluation Plot with Confidence Intervals
    print("\n📈  Generating Aggregate Evaluation Plot...")

    aggregate_eval_path = os.path.join(plots_dir, f"aggregate_testing_evaluation_{suffix}.png")
    # Note: We now pass the lists directly instead of the flattened arrays
    plot_aggregate_testing_evaluation(global_y_test_list, global_y_pred_prob_list, global_y_pred_class_list,
                                      aggregate_eval_path)
    print(f"✅  Global testing evaluation plot saved to: {aggregate_eval_path}")

    primary_plot_path = os.path.join(plots_dir, f"aggregate_primary_metrics_{suffix}.png")
    plot_primary_metrics(summary_df, primary_plot_path)
    print(f"✅  Primary aggregate plot saved to: {primary_plot_path}")

    secondary_plot_path = os.path.join(plots_dir, f"aggregate_secondary_metrics_{suffix}.png")
    plot_secondary_metrics(summary_df, secondary_plot_path, DECISION_THRESHOLD)
    print(f"✅  Secondary aggregate plot saved to: {secondary_plot_path}")

    avg_history_df = calculate_average_history(all_histories)
    if avg_history_df is not None:
        avg_history_path = os.path.join(plots_dir, f"aggregate_training_history_{suffix}.png")
        plot_average_training_history(avg_history_df, avg_history_path)
        print(f"✅  Average training history plot saved to: {avg_history_path}")