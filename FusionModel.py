# %% Imports and Configuration
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from IPython.display import display

# =============================================================================
# 🚀 ENSEMBLE CONFIGURATION
# =============================================================================
BATCH_SIZE = 128
DECISION_THRESHOLD = 0.3
VOTING_METHOD = 'hard'  # Choose 'soft' or 'hard'

# Define the models and their corresponding datasets to ensemble
ENSEMBLE_CONFIG = {
    'master_dataset_1s.npz': 'saved_outputs/20260304-135510-1s/models',
    'master_dataset_2s.npz': 'saved_outputs/20260303-162053-2s/models',
    'master_dataset_4s.npz': 'saved_outputs/20260304-091718-4s/models',
    'master_dataset_5s.npz': 'saved_outputs/20260309-115134_5s/models',
    'master_dataset_10s.npz': 'saved_outputs/20260309-115134_5s/models',
}

output_dir = os.path.join("saved_outputs", "ensemble_results")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# %% Data Loading
# =============================================================================
print("Loading datasets for ensemble...")
datasets = {}
for ds_path in ENSEMBLE_CONFIG.keys():
    data = np.load(ds_path)
    datasets[ds_path] = {'X': data['X'], 'y': data['y'], 'groups': data['s']}
    print(f"Loaded {ds_path} | Shape: {data['X'].shape}")

# Designate the first dataset (1s) as the base resolution for alignment
base_ds_name = list(ENSEMBLE_CONFIG.keys())[0]
groups_base = datasets[base_ds_name]['groups']
y_base = datasets[base_ds_name]['y']
logo = LeaveOneGroupOut()

# =============================================================================
# %% LOSO Ensemble Evaluation Loop
# =============================================================================
all_reports = []

# Split based on the BASE dataset
for train_idx_base, test_idx_base in logo.split(datasets[base_ds_name]['X'], y_base, groups=groups_base):
    current_test_subject = groups_base[test_idx_base][0]

    print(f"\n{'=' * 60}")
    print(f"🧪  Evaluating Ensemble for Subject: {current_test_subject}")
    print(f"{'=' * 60}")

    subject_predictions_prob = []

    # The ground truth is taken STRICTLY from the base dataset (1s)
    y_test_actual = datasets[base_ds_name]['y'][test_idx_base]
    target_alignment_length = len(y_test_actual)

    valid_ensemble = True

    # Iterate through each model/dataset in the ensemble
    for ds_path, models_dir in ENSEMBLE_CONFIG.items():
        # 1. Find the test indices for the current subject in THIS specific dataset
        # (Since lengths differ, we cannot use test_idx_base for the 2s and 4s datasets)
        subject_mask = (datasets[ds_path]['groups'] == current_test_subject)
        X_test_ds = datasets[ds_path]['X'][subject_mask]

        model_path = os.path.join(models_dir, f"best_model_subject_{current_test_subject}.keras")

        if not os.path.exists(model_path):
            print(f"⚠️ Skipping Subject {current_test_subject} — missing model for {ds_path}")
            valid_ensemble = False
            break

        # 2. Load model and predict raw probabilities
        model = tf.keras.models.load_model(model_path)
        y_pred_prob_raw = model.predict(X_test_ds, batch_size=BATCH_SIZE, verbose=0).ravel()

        # 3. TEMPORAL ALIGNMENT (Interpolation)
        orig_len = len(y_pred_prob_raw)

        if orig_len == target_alignment_length:
            aligned_prob = y_pred_prob_raw  # 1s dataset needs no interpolation
        else:
            # Map both arrays to a normalized 0.0 to 1.0 time scale
            orig_time_axis = np.linspace(0, 1, orig_len)
            target_time_axis = np.linspace(0, 1, target_alignment_length)

            # Interpolate the probabilities to match the base dataset's length
            aligned_prob = np.interp(target_time_axis, orig_time_axis, y_pred_prob_raw)
            print(f"    Interpolated {ds_path} predictions from {orig_len} to {target_alignment_length} steps.")

        subject_predictions_prob.append(aligned_prob)

        del model
        tf.keras.backend.clear_session()

    if not valid_ensemble:
        continue  # Skip this subject if any model is missing

    # Convert the list of arrays into a single 2D NumPy array (Shape: [Num_Models, Target_Length])
    subject_predictions_prob = np.array(subject_predictions_prob)

    # Generate discrete classes from the aligned probabilities
    subject_predictions_class = (subject_predictions_prob >= DECISION_THRESHOLD).astype(int)

    # =========================================================================
    # FUSION LOGIC (Soft vs. Hard Voting)
    # =========================================================================
    if VOTING_METHOD == 'soft':
        # Average the interpolated probabilities across all models
        ensemble_prob = np.mean(subject_predictions_prob, axis=0)
        ensemble_class = (ensemble_prob >= DECISION_THRESHOLD).astype(int)

    elif VOTING_METHOD == 'hard':
        # Majority vote on the discrete classes
        class_sum = np.sum(subject_predictions_class, axis=0)
        ensemble_class = (class_sum > (len(ENSEMBLE_CONFIG) / 2)).astype(int)
        ensemble_prob = class_sum / len(ENSEMBLE_CONFIG)

    else:
        raise ValueError("VOTING_METHOD must be 'soft' or 'hard'")

    # =========================================================================
    # METRICS CALCULATION
    # =========================================================================
    auroc = roc_auc_score(y_test_actual, ensemble_prob)
    auprc = average_precision_score(y_test_actual, ensemble_prob)

    report_dict = classification_report(
        y_test_actual, ensemble_class,
        target_names=["Normal", "Seizure"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test_actual, ensemble_class)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_test_actual, ensemble_class)
    seizure_f1 = report_dict["Seizure"]["f1-score"]

    print(f"  Fusion Type          = {VOTING_METHOD.capitalize()} Voting")
    print(f"  Ensemble AUROC       = {auroc:.4f}")
    print(f"  Ensemble AUPRC       = {auprc:.4f}")
    print(f"  Sensitivity (Recall) = {sensitivity:.4f}")
    print(f"  Specificity          = {specificity:.4f}")
    print(f"  Balanced Acc         = {balanced_acc:.4f}")

    all_reports.append({
        "subject": current_test_subject,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": report_dict["accuracy"],
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "seizure_f1": seizure_f1,
    })

# =============================================================================
# %% Aggregate Results Summary
# =============================================================================
if all_reports:
    summary_df = pd.DataFrame(all_reports)

    print(f"\n{'=' * 60}")
    print(f"🏆  Overall LOSO Ensemble Summary ({VOTING_METHOD.capitalize()} Voting)")
    print(f"{'=' * 60}")

    summary_csv_path = os.path.join(output_dir, f"ensemble_{VOTING_METHOD}_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n✅  Summary saved to: {summary_csv_path}")

    metrics = ["auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "seizure_f1"]
    col_width = 22
    print(f"\n📊  Aggregate Statistics (mean ± std across subjects):")
    print(f"  {'Metric':<{col_width}} {'Mean':>8}   {'Std':>8}")
    print(f"  {'-' * 45}")
    for m in metrics:
        print(f"  {m:<{col_width}} {summary_df[m].mean():>8.4f}   {summary_df[m].std():>8.4f}")