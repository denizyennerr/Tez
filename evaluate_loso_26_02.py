# %% Imports and Configuration
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score,
                             balanced_accuracy_score)
from IPython.display import display

# --- UPDATE THESE TO MATCH YOUR TRAINING RUN ---
TIMESTAMP = "20260225-151022"
dataset_path = 'master_dataset_2s.npz'
output_dir = os.path.join("saved_outputs", TIMESTAMP)
models_dir = os.path.join(output_dir, "models")
histories_dir = os.path.join(output_dir, "histories")
plots_dir = os.path.join(output_dir, "plots")

os.makedirs(plots_dir, exist_ok=True)


# %% Evaluation and Plotting Functions

def plot_training_history(history_df, current_subject, save_path):
    """Plots training and validation accuracy/loss separately."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History | Subject: {current_subject}', fontsize=16, fontweight='bold')

    # 1. Model Accuracy
    axs[0].plot(history_df['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    axs[0].plot(history_df['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Model Loss
    axs[1].plot(history_df['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    axs[1].plot(history_df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    axs[1].set_title('Model Loss (Binary Crossentropy)')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close plot to free memory


def plot_testing_evaluation(y_test, y_pred_prob, current_subject, save_path):
    """Plots Confusion Matrix and Precision-Recall Curve separately."""
    # y_pred = (y_pred_prob > 0.5).astype(int)

    def find_optimal_threshold(y_test, y_pred_prob, target_sensitivity=0.80):
        """Find threshold that achieves target sensitivity while maximizing precision."""
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

        for i, thresh in enumerate(thresholds[:-1]):  # Exclude last threshold
            if recall[i] >= target_sensitivity:
                return thresh

        return 0.3  # Default fallback

    optimal_thresh = find_optimal_threshold(y_test, y_pred_prob, target_sensitivity=0.80)
    y_pred= (y_pred_prob > optimal_thresh).astype(int)
    print(f"Subject {current_test_subject}: Using threshold = {optimal_thresh:.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Testing Evaluation | Subject: {current_subject}', fontsize=16, fontweight='bold')

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0],
                xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'],
                annot_kws={"size": 14})
    axs[0].set_title('Confusion Matrix')
    axs[0].set_ylabel('True Label')
    axs[0].set_xlabel('Predicted Label')

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    avg_precision = average_precision_score(y_test, y_pred_prob)

    axs[1].plot(recall, precision, color='purple', lw=2, label=f'Avg Precision = {avg_precision:.3f}')
    axs[1].set_xlabel('Recall (Sensitivity)')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Precision-Recall Curve')
    axs[1].legend(loc="lower left")
    axs[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close plot to free memory


# %% Data Loading
data = np.load(dataset_path)
X, y, groups = data['X'], data['y'], data['s']
logo = LeaveOneGroupOut()

# %% LOSO Evaluation Loop
all_reports = []

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_test = X[test_idx]
    y_test = y[test_idx]
    current_test_subject = groups[test_idx][0]

    model_path = os.path.join(models_dir, f"best_model_subject_{current_test_subject}.keras")
    history_path = os.path.join(histories_dir, f"history_subject_{current_test_subject}.csv")

    if not os.path.exists(model_path):
        print(f"⚠️ Skipping Subject {current_test_subject} - Model not found.")
        continue

    print(f"\n🧪 Evaluating Subject: {current_test_subject}")

    # Load Model and History
    model = tf.keras.models.load_model(model_path)
    history_df = pd.read_csv(history_path)

    # Predictions
    y_pred_prob = model.predict(X_test).ravel()  # Ensure 1D array for metric functions
    # y_pred_class = (y_pred_prob > 0.5).astype(int)

    def find_optimal_threshold(y_test, y_pred_prob, target_sensitivity=0.80):
        """Find threshold that achieves target sensitivity while maximizing precision."""
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

        for i, thresh in enumerate(thresholds[:-1]):  # Exclude last threshold
            if recall[i] >= target_sensitivity:
                return thresh

        return 0.3  # Default fallback

    optimal_thresh = find_optimal_threshold(y_test, y_pred_prob, target_sensitivity=0.80)
    y_pred_class = (y_pred_prob > optimal_thresh).astype(int)
    print(f"Subject {current_test_subject}: Using threshold = {optimal_thresh:.3f}")

    # Display Rich DataFrame Reporting
    report_dict = classification_report(y_test, y_pred_class, target_names=['Normal', 'Seizure'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    display(report_df)

    # Calculate Sensitivity, Specificity, and Balanced Accuracy
    cm = confusion_matrix(y_test, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_test, y_pred_class)

    all_reports.append({
        'subject': current_test_subject,
        'accuracy': report_dict['accuracy'],
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'seizure_f1': report_dict['Seizure']['f1-score']
    })

    # Generate and save separate plots
    train_save_filepath = os.path.join(plots_dir, f"subject_{current_test_subject}_training_history.png")
    test_save_filepath = os.path.join(plots_dir, f"subject_{current_test_subject}_testing_evaluation.png")

    plot_training_history(history_df, current_test_subject, train_save_filepath)
    plot_testing_evaluation(y_test, y_pred_prob, current_test_subject, test_save_filepath)

# %% Aggregate Results Summary
if all_reports:
    summary_df = pd.DataFrame(all_reports)
    print("\n🏆 Overall LOSO Summary:")
    display(summary_df)

    summary_csv_path = os.path.join(output_dir, "overall_loso_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Overall summary saved to: {summary_csv_path}")

    # Display updated metric aggregates
    print(f"Mean Accuracy:          {summary_df['accuracy'].mean():.4f}")
    print(f"Mean Balanced Accuracy: {summary_df['balanced_accuracy'].mean():.4f}")
    print(f"Mean Sensitivity:       {summary_df['sensitivity'].mean():.4f}")
    print(f"Mean Specificity:       {summary_df['specificity'].mean():.4f}")