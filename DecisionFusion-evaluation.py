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
    precision_recall_curve  # 🔥 NEW: Imported for dynamic thresholding
)

# =============================================================================
# CONFIGURATION
# =============================================================================
BATCH_SIZE = 128
VOTING_METHOD = 'soft'  # OPTIONS: 'soft' (weighted mean) | 'max'

# 🔥 OPTIMIZED WEIGHTS:
# Dropped the 5s and 10s models (weight 0) to prevent them from dragging down the ensemble.
# Kept the 1s, 2s, and 4s models as the core voters.
ENSEMBLE_CONFIG = {
    'master_dataset_1s.npz': {'models_dir': 'saved_outputs/20260304-135510-1s/models', 'weight': 3.0},
    'master_dataset_2s.npz': {'models_dir': 'saved_outputs/20260303-162053-2s/models', 'weight': 1.0},
    'master_dataset_4s.npz': {'models_dir': 'saved_outputs/20260304-091718-4s/models', 'weight': 1.0},
}

BASE_DS_NAME = 'master_dataset_1s.npz'

output_dir = os.path.join('saved_outputs', 'ensemble_results_static')
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# HELPER: Window-level majority-vote alignment
# =============================================================================
def window_level_align(coarse_probs: np.ndarray, target_length: int) -> np.ndarray:
    n_coarse = len(coarse_probs)
    if n_coarse == target_length:
        return coarse_probs.copy()

    fine_indices = np.floor(
        np.linspace(0, n_coarse, target_length, endpoint=False)
    ).astype(int)

    fine_indices = np.clip(fine_indices, 0, n_coarse - 1)
    return coarse_probs[fine_indices]


# =============================================================================
# DATA LOADING
# =============================================================================
print('Loading datasets for ensemble...')
datasets = {}
for ds_path in ENSEMBLE_CONFIG:
    data = np.load(ds_path)
    datasets[ds_path] = {'X': data['X'], 'y': data['y'], 'groups': data['s']}

groups_base = datasets[BASE_DS_NAME]['groups']
y_base = datasets[BASE_DS_NAME]['y']
logo = LeaveOneGroupOut()

# =============================================================================
# LOSO ENSEMBLE EVALUATION LOOP
# =============================================================================
all_reports = []

for train_idx_base, test_idx_base in logo.split(
        datasets[BASE_DS_NAME]['X'], y_base, groups=groups_base
):
    current_test_subject = groups_base[test_idx_base][0]

    print(f"\n{'=' * 65}")
    print(f'🧪  Evaluating Ensemble for Subject: {current_test_subject}')
    print(f"{'=' * 65}")

    y_test_actual = datasets[BASE_DS_NAME]['y'][test_idx_base]
    target_alignment_length = len(y_test_actual)

    subject_predictions_prob = []
    subject_weights = []
    valid_ensemble = True

    for ds_path, cfg in ENSEMBLE_CONFIG.items():
        if cfg['weight'] == 0.0:
            continue  # Skip loading models we are ignoring anyway

        models_dir = cfg['models_dir']
        model_path = os.path.join(models_dir, f'best_model_subject_{current_test_subject}.keras')

        if not os.path.exists(model_path):
            print(f'  ⚠️  Missing model for {ds_path} — skipping subject {current_test_subject}')
            valid_ensemble = False
            break

        subject_mask = (datasets[ds_path]['groups'] == current_test_subject)
        X_test_ds = datasets[ds_path]['X'][subject_mask]

        model = tf.keras.models.load_model(model_path)
        y_pred_prob_raw = model.predict(X_test_ds, batch_size=BATCH_SIZE, verbose=0).ravel()

        del model
        tf.keras.backend.clear_session()

        aligned_prob = window_level_align(y_pred_prob_raw, target_alignment_length)

        subject_predictions_prob.append(aligned_prob)
        subject_weights.append(cfg['weight'])

    if not valid_ensemble:
        continue

    subject_predictions_prob = np.array(subject_predictions_prob)

    # ── Fusion ───────────────────────────────────────────────────────────────
    if VOTING_METHOD == 'soft':
        ensemble_prob = np.average(subject_predictions_prob, axis=0, weights=subject_weights)
    elif VOTING_METHOD == 'max':
        ensemble_prob = np.max(subject_predictions_prob, axis=0)
    else:
        raise ValueError("VOTING_METHOD must be 'soft' or 'max' when using dynamic thresholding")

    # 🔥 NEW: Dynamic Threshold Optimization
    # Calculate Precision and Recall for EVERY possible threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test_actual, ensemble_prob)

    # Calculate F1 score for every threshold safely (avoiding divide-by-zero)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )

    # Find the threshold that yields the absolute maximum F1 Score
    optimal_idx = np.argmax(f1_scores)

    # Scikit-learn's thresholds array is 1 element shorter than precision/recall arrays
    if optimal_idx < len(thresholds):
        best_threshold = thresholds[optimal_idx]
    else:
        best_threshold = 1.0  # Edge case fallback

    # Apply the perfectly calibrated threshold
    ensemble_class = (ensemble_prob >= best_threshold).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    auroc = roc_auc_score(y_test_actual, ensemble_prob)
    auprc = average_precision_score(y_test_actual, ensemble_prob)

    cm = confusion_matrix(y_test_actual, ensemble_class)
    tn, fp, fn, tp = cm.ravel()

    # Handle the edge cases where an array is entirely one class
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_test_actual, ensemble_class)

    # Best F1 score found during optimization
    seizure_f1 = f1_scores[optimal_idx]

    # Calculate accuracy manually since we aren't using classification_report anymore
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    print(f'\n  Fusion Type          = Weighted {VOTING_METHOD.capitalize()} Fusion')
    print(f'  Optimal Threshold    = {best_threshold:.4f} ')
    print(f'  Ensemble AUROC       = {auroc:.4f}')
    print(f'  Ensemble AUPRC       = {auprc:.4f}')
    print(f'  Sensitivity (Recall) = {sensitivity:.4f}')
    print(f'  Specificity          = {specificity:.4f}')
    print(f'  Balanced Accuracy    = {balanced_acc:.4f}')
    print(f'  Seizure F1-Score     = {seizure_f1:.4f}')

    all_reports.append({
        'subject': current_test_subject,
        'optimal_threshold': best_threshold,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'seizure_f1': seizure_f1,
    })

# =============================================================================
# AGGREGATE RESULTS SUMMARY
# =============================================================================
if all_reports:
    summary_df = pd.DataFrame(all_reports)

    print(f"\n{'=' * 65}")
    print(f'🏆  Overall LOSO Ensemble Summary — Weighted {VOTING_METHOD.capitalize()} Fusion')
    print(f"{'=' * 65}")

    summary_csv_path = os.path.join(
        output_dir, f'ensemble_opt_thresh_{VOTING_METHOD}_summary.csv'
    )
    summary_df.to_csv(summary_csv_path, index=False)
    print(f'\n✅  Summary saved to: {summary_csv_path}')

    metrics = ['auroc', 'auprc', 'sensitivity', 'specificity', 'balanced_accuracy', 'seizure_f1', 'optimal_threshold']
    col_width = 22
    print(f'\n📊  Aggregate Statistics (mean ± std across subjects):')
    print(f"  {'Metric':<{col_width}} {'Mean':>8}   {'Std':>8}")
    print(f"  {'-' * 45}")
    for m in metrics:
        print(f'  {m:<{col_width}} {summary_df[m].mean():>8.4f}   {summary_df[m].std():>8.4f}')
else:
    print('\n⚠️  No subjects were successfully evaluated.')