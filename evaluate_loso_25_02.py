# %% Imports and Configuration
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from IPython.display import display

# --- UPDATE THESE TO MATCH YOUR TRAINING RUN ---
TIMESTAMP = "_25-02"
dataset_path = 'master_dataset_2s.npz'
output_dir = os.path.join("saved_outputs", TIMESTAMP)
models_dir = os.path.join(output_dir, "models")
histories_dir = os.path.join(output_dir, "histories")
plots_dir = os.path.join(output_dir, "plots")

os.makedirs(plots_dir, exist_ok=True)


# %% Evaluation and Plotting Functions
def plot_fold_diagnostics(history_df, y_test, y_pred_prob, current_subject, save_path):
    """Generates a 2x2 dashboard, expecting a Pandas DataFrame for history."""
    y_pred = (y_pred_prob > 0.5).astype(int)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Dashboard | Subject: {current_subject}', fontsize=16, fontweight='bold')

    # 1. Model Accuracy (Now reading from DataFrame)
    axs[0, 0].plot(history_df['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    axs[0, 0].plot(history_df['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    axs[0, 0].set_title('Model Accuracy')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Model Loss (Now reading from DataFrame)
    axs[0, 1].plot(history_df['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    axs[0, 1].plot(history_df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    axs[0, 1].set_title('Model Loss (Binary Crossentropy)')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1, 0],
                xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'],
                annot_kws={"size": 14})
    axs[1, 0].set_title('Confusion Matrix')
    axs[1, 0].set_ylabel('True Label')
    axs[1, 0].set_xlabel('Predicted Label')

    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].set_title('Receiver Operating Characteristic (ROC)')
    axs[1, 1].legend(loc="lower right")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


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
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    # Rich DataFrame Reporting
    report_dict = classification_report(y_test, y_pred_class, target_names=['Normal', 'Seizure'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    display(report_df)

    all_reports.append({
        'subject': current_test_subject,
        'accuracy': report_dict['accuracy'],
        'seizure_f1': report_dict['Seizure']['f1-score'],
        'seizure_recall': report_dict['Seizure']['recall']
    })

    # Generate and save plots
    save_filepath = os.path.join(plots_dir, f"subject_{current_test_subject}_dashboard.png")
    plot_fold_diagnostics(history_df, y_test, y_pred_prob, current_test_subject, save_filepath)

# %% Aggregate Results Summary
if all_reports:
    summary_df = pd.DataFrame(all_reports)
    print("\n🏆 Overall LOSO Summary:")
    display(summary_df)

    summary_csv_path = os.path.join(output_dir, "overall_loso_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Overall summary saved to: {summary_csv_path}")

    print(f"Mean Accuracy: {summary_df['accuracy'].mean():.4f}")
    print(f"Mean Seizure Sensitivity (Recall): {summary_df['seizure_recall'].mean():.4f}")