import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================
datasets_info = {
    '0.5s': {'data': 'master_dataset_0.5s.npz', 'models_dir': 'saved_outputs/20260311-144059_0.5s/models'},
    '1s': {'data': 'master_dataset_1s.npz', 'models_dir': 'saved_outputs/20260304-135510-1s/models'},
    '2s': {'data': 'master_dataset_2s.npz', 'models_dir': 'saved_outputs/20260303-162053-2s/models'},
    '4s': {'data': 'master_dataset_4s.npz', 'models_dir': 'saved_outputs/20260304-091718-4s/models'},
    '5s': {'data': 'master_dataset_5s.npz', 'models_dir': 'saved_outputs/20260309-115134_5s/models'},
    '10s': {'data': 'master_dataset_10s.npz', 'models_dir': 'saved_outputs/20260309-155522_10s/models'},
}

output_csv_name = "all_models_performance_comparison_1s_ref_0.5tresh.csv"

# Weights based on Eval script insights
model_weights = {'0.5s': 1.0, '1s': 3.0, '2s': 1.0, '4s': 1.0, '5s': 0.0, '10s': 0.0}

# FIXED A PRIORI CLINICAL THRESHOLD
FIXED_THRESHOLD = 0.5


# =============================================================================
# 2. CORE FUNCTIONS
# =============================================================================
def window_level_align(coarse_probs: np.ndarray, target_length: int) -> np.ndarray:
    """Robust alignment of probabilities to a target length using index interpolation."""
    n_coarse = len(coarse_probs)
    if n_coarse == target_length:
        return coarse_probs.copy()

    fine_indices = np.floor(
        np.linspace(0, n_coarse, target_length, endpoint=False)
    ).astype(int)

    fine_indices = np.clip(fine_indices, 0, n_coarse - 1)
    return coarse_probs[fine_indices]


def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    # Handle single-class edge cases
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc, auprc = np.nan, np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        'Decision Threshold': round(threshold, 4),
        'AUROC': round(auroc, 4),
        'AUPRC': round(auprc, 4),
        'Sensitivity': round(sensitivity, 4),
        'Specificity': round(specificity, 4),
        'Balanced Accuracy': round(balanced_acc, 4),
        'Seizure F1-Score': round(f1_score, 4)
    }


# =============================================================================
# 3. DATA LOADING & PROBABILITY ALIGNMENT
# =============================================================================
all_y_true_1s = []
all_probs_dict = {k: [] for k in datasets_info.keys()}

print("Loading 1s reference data to determine subjects...")
ref_data = np.load(datasets_info['1s']['data'])
subjects = np.unique(ref_data['s'])
del ref_data;
gc.collect()

for subject in subjects:
    print(f"\n🚀 Processing Subject {subject}...")

    # Load 1s target data to determine the target sequence length
    data_1s = np.load(datasets_info['1s']['data'])
    mask_1s = (data_1s['s'] == subject)
    y_test_1s = data_1s['y'][mask_1s]
    target_len = len(y_test_1s)
    all_y_true_1s.extend(y_test_1s)
    del data_1s;
    gc.collect()

    for w_name, info in datasets_info.items():
        data = np.load(info['data'])
        mask = (data['s'] == subject)
        X_test = data['X'][mask]

        model_path = os.path.join(info['models_dir'], f"best_model_subject_{subject}.keras")

        if os.path.exists(model_path):
            model = load_model(model_path)
            probs = model.predict(X_test, batch_size=64, verbose=0).ravel()

            aligned_probs = window_level_align(probs, target_len)
            all_probs_dict[w_name].extend(aligned_probs)

            del model;
            K.clear_session()
        else:
            print(f"  ⚠️ Model missing: {w_name} - Subject {subject}")
            # Fallback array
            all_probs_dict[w_name].extend(np.full(target_len, 0.5))

        del data, X_test;
        gc.collect()

# =============================================================================
# 4. DECISION FUSION & EVALUATION (STRICTLY RIGOROUS)
# =============================================================================
print(f"\n{'=' * 65}\nEvaluating Models & Ensembles (Threshold: {FIXED_THRESHOLD})\n{'=' * 65}")

y_true = np.array(all_y_true_1s)
results = []

# --- Individual Models ---
for w_name in datasets_info.keys():
    probs = np.array(all_probs_dict[w_name])
    res = calculate_metrics(y_true, probs, threshold=FIXED_THRESHOLD)
    res['Model'] = f"{w_name} Individual"
    results.append(res)

# --- Soft Ensemble (Simple Mean) ---
soft_probs = np.mean([all_probs_dict[k] for k in datasets_info.keys()], axis=0)
res_soft = calculate_metrics(y_true, soft_probs, threshold=FIXED_THRESHOLD)
res_soft['Model'] = "Ensemble (Soft Mean)"
results.append(res_soft)

# --- Weighted Ensemble ---
weighted_probs = np.zeros_like(y_true, dtype=float)
active_weights_sum = sum(model_weights.values())

for k in datasets_info.keys():
    weighted_probs += np.array(all_probs_dict[k]) * (model_weights[k] / active_weights_sum)

res_weight = calculate_metrics(y_true, weighted_probs, threshold=FIXED_THRESHOLD)
res_weight['Model'] = "Ensemble (Weighted Soft)"
results.append(res_weight)

# --- Hard Ensemble (Majority Vote) ---
hard_votes = np.zeros_like(y_true, dtype=int)
for k in datasets_info.keys():
    p = np.array(all_probs_dict[k])
    # Individual models cast their vote based on the clinical priority threshold of 0.3
    hard_votes += (p >= FIXED_THRESHOLD).astype(int)

# Normalize votes to pseudo-probabilities for the calculate_metrics function
hard_probs = hard_votes / len(datasets_info)
# The final ensemble threshold is rigidly 0.5 (representing a > 50% majority agreement)
res_hard = calculate_metrics(y_true, hard_probs, threshold=0.5)
res_hard['Model'] = "Ensemble (Hard Vote)"
results.append(res_hard)

# =============================================================================
# 5. EXPORT & SUMMARY
# =============================================================================
columns_order = [
    'Model', 'Decision Threshold', 'AUROC', 'AUPRC',
    'Sensitivity', 'Specificity', 'Balanced Accuracy', 'Seizure F1-Score'
]
df_results = pd.DataFrame(results)[columns_order]
df_results.to_csv(output_csv_name, index=False)

print(f"\n✅ Results saved to {output_csv_name}")
print("\n" + df_results.to_string(index=False))