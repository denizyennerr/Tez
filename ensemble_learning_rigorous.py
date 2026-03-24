import os
import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from scipy.signal import medfilt
import tensorflow as tf


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class DatasetInfo:
    data: str
    models_dir: str


DATASETS_INFO: Dict[str, DatasetInfo] = {
    '0.5s': DatasetInfo(
        data='processed_master_datasets/master_dataset_0.5s.npz',
        models_dir='saved_outputs_play/20260323-163632_0.5s/models'),
    '1.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_1.0s.npz',
        models_dir='saved_outputs_play/20260323-201436_1.0s/models'),
    '2.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_2.0s.npz',
        models_dir='saved_outputs_play/20260323-220314_2.0s/models'),
    '4.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_4.0s.npz',
        models_dir='saved_outputs_play/20260323-224227_4.0s/models'),
    '5.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_5.0s.npz',
        models_dir='saved_outputs_play/20260323-231310_5.0s/models'),
    '10.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_10.0s.npz',
        models_dir='saved_outputs_play/20260323-234451_10.0s/models'),
}

REFERENCE_KEY = '0.5s'
BATCH_SIZE = 128

# Split data for threshold/weight optimization vs testing
VAL_SPLIT_RATIO = 0.30
TARGET_SENSITIVITY = 0.80
MIN_PRECISION_FLOOR = 0.10
FIXED_THRESHOLD = 0.35
MIN_MODELS_FOR_ENSEMBLE = 2
MEDIAN_FILTER_WINDOW = 5

OUTPUT_DIR = os.path.join('saved_outputs_play', 'ensemble_results_rigorous')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_POOLED_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_pooled_metrics.csv')
OUTPUT_MACRO_MEAN_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_macro_mean.csv')
OUTPUT_MACRO_STD_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_macro_std.csv')
OUTPUT_PER_SUBJECT_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_per_subject.csv')
OUTPUT_THRESHOLDS_WEIGHTS_CSV = os.path.join(OUTPUT_DIR, 'subject_optimal_params.csv')

# =============================================================================
# 2. UTILITIES
# =============================================================================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def window_level_align(coarse_probs: np.ndarray, target_length: int) -> np.ndarray:
    n_coarse = len(coarse_probs)
    if n_coarse == target_length:
        return coarse_probs.copy()
    if target_length % n_coarse == 0:
        return np.repeat(coarse_probs, target_length // n_coarse)

    fine_indices = np.floor(np.linspace(0, n_coarse, target_length, endpoint=False)).astype(int)
    return coarse_probs[np.clip(fine_indices, 0, n_coarse - 1)]


def find_sensitivity_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return FIXED_THRESHOLD

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]

    both_ok = (recalls >= TARGET_SENSITIVITY) & (precisions >= MIN_PRECISION_FLOOR)
    if np.any(both_ok):
        return float(thresholds[both_ok][np.argmax(precisions[both_ok])])

    sens_ok = recalls >= TARGET_SENSITIVITY
    if np.any(sens_ok):
        return float(thresholds[sens_ok][np.argmax(precisions[sens_ok])])

    return FIXED_THRESHOLD


def temporal_smooth_probs(probs: np.ndarray, window: int = MEDIAN_FILTER_WINDOW) -> np.ndarray:
    if window <= 1:
        return probs.copy()
    if window % 2 == 0:
        window += 1
    return medfilt(probs.astype(float), kernel_size=window)


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    has_positives = len(np.unique(y_true)) > 1

    auroc = roc_auc_score(y_true, y_prob) if has_positives else np.nan
    auprc = average_precision_score(y_true, y_prob) if has_positives else np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0


    return {
        'Decision Threshold': round(float(threshold), 4),
        'AUROC': round(float(auroc), 4),
        'AUPRC': round(float(auprc), 4),
        'Sensitivity': round(float(sensitivity), 4),
        'Specificity': round(float(specificity), 4),
        'Balanced Accuracy': round(float(balanced_acc), 4),
        'Precision': round(float(precision), 4),
        'Seizure F1-Score': round(float(f1), 4),
    }


def optimize_ensemble_weights(probs_list: List[np.ndarray], y_true: np.ndarray) -> List[float]:
    """Uses Logistic Regression to find the optimal contribution of each temporal resolution."""
    if len(np.unique(y_true)) < 2:
        return [1.0 / len(probs_list)] * len(probs_list)

    X = np.column_stack(probs_list)

    # Removed 'positive=True' to fix the TypeError
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X, y_true)

    # Manually clip negative weights to 0 to ensure logical ensemble voting
    weights = np.clip(clf.coef_[0], 0, None)

    # Fallback to equal weighting if all weights become 0
    if np.sum(weights) == 0:
        return [1.0 / len(probs_list)] * len(probs_list)

    return (weights / np.sum(weights)).tolist()  # Normalize to sum to 1


# =============================================================================
# 3. CORE LOGIC (Split & Evaluate)
# =============================================================================

def soft_ensemble(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(probs_list), axis=0)


def weighted_soft_ensemble(probs_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    return np.average(np.vstack(probs_list), axis=0, weights=np.asarray(weights, dtype=float))


def hard_vote_ensemble(probs_list: List[np.ndarray], threshold: float = FIXED_THRESHOLD) -> np.ndarray:
    return np.mean((np.vstack(probs_list) >= threshold).astype(int), axis=0)


def main() -> None:
    # 1. Load Reference Data
    ref_data = np.load(DATASETS_INFO[REFERENCE_KEY].data)
    subjects = np.unique(ref_data['s'])

    # Dictionaries to hold validation (train) and test splits
    y_val_dict, y_test_dict = {}, {}
    p_val_dict, p_test_dict = {k: {} for k in DATASETS_INFO.keys()}, {k: {} for k in DATASETS_INFO.keys()}

    for subject in subjects:
        mask = ref_data['s'] == subject
        y_sub = ref_data['y'][mask]

        split_idx = int(len(y_sub) * VAL_SPLIT_RATIO)
        y_val_dict[str(subject)] = y_sub[:split_idx]
        y_test_dict[str(subject)] = y_sub[split_idx:]

    # 2. Collect and Split Probabilities
    for res_key, info in DATASETS_INFO.items():
        if not os.path.exists(info.data): continue
        data = np.load(info.data)
        X, s = np.swapaxes(data['X'], 0, 1), data['s']

        for subject in subjects:
            sub_str = str(subject)
            model_path = os.path.join(info.models_dir, f'best_model_subject_{sub_str}.keras')
            if not os.path.exists(model_path): continue

            mask = s == subject
            if not np.any(mask): continue

            # Predict
            model = load_model(model_path, compile=False)
            probs_raw = model.predict(tf.data.Dataset.from_tensor_slices(X[mask]).batch(BATCH_SIZE), verbose=0).ravel()
            K.clear_session()

            aligned = window_level_align(probs_raw, len(ref_data['y'][ref_data['s'] == subject]))

            # Split chronologically to avoid data leakage
            split_idx = int(len(aligned) * VAL_SPLIT_RATIO)
            p_val_dict[res_key][sub_str] = aligned[:split_idx]
            p_test_dict[res_key][sub_str] = aligned[split_idx:]

    del ref_data, data;
    gc.collect()

    # 3. Optimize Weights and Thresholds on Validation Data ONLY
    logging.info('Optimizing weights and thresholds on chronologically split validation data...')
    subject_params = {}

    for subject in subjects:
        sub_str = str(subject)
        val_probs_list = [p_val_dict[k].get(sub_str) for k in DATASETS_INFO.keys() if
                          p_val_dict[k].get(sub_str) is not None]

        if len(val_probs_list) < MIN_MODELS_FOR_ENSEMBLE: continue

        # Optimize Weights via Logistic Regression
        opt_weights = optimize_ensemble_weights(val_probs_list, y_val_dict[sub_str])

        # Optimize Threshold via Sensitivity Check on smoothed Weighted Ensemble
        weighted_val_probs = weighted_soft_ensemble(val_probs_list, opt_weights)
        smoothed_val_probs = temporal_smooth_probs(weighted_val_probs)
        opt_threshold = find_sensitivity_optimal_threshold(y_val_dict[sub_str], smoothed_val_probs)

        subject_params[sub_str] = {'weights': opt_weights, 'threshold': opt_threshold}

    # Save learned parameters
    pd.DataFrame([{'Subject': s, 'Threshold': p['threshold'], 'Weights': p['weights']}
                  for s, p in subject_params.items()]).to_csv(OUTPUT_THRESHOLDS_WEIGHTS_CSV, index=False)

    # 4. Evaluate on Test Data ONLY
    logging.info('Evaluating strictly on unseen test holdouts...')
    results = []

    for subject in subjects:
        sub_str = str(subject)
        if sub_str not in subject_params: continue

        y_test = y_test_dict[sub_str]
        test_probs_list = [p_test_dict[k].get(sub_str) for k in DATASETS_INFO.keys() if
                           p_test_dict[k].get(sub_str) is not None]
        weights = subject_params[sub_str]['weights']
        opt_thr = subject_params[sub_str]['threshold']

        # -------- Individual Epoch Length Baselines --------
        for res_key in DATASETS_INFO.keys():
            prob = p_test_dict[res_key].get(sub_str)
            if prob is not None:
                results.append({
                    **calculate_metrics(y_test, prob, FIXED_THRESHOLD),
                    'Model': f'{res_key} Individual',
                    'Subject': sub_str
                })

        # -------- Ensemble Baselines --------
        soft_p = soft_ensemble(test_probs_list)
        results.append(
            {**calculate_metrics(y_test, soft_p, FIXED_THRESHOLD), 'Model': 'Ensemble (Soft Mean)', 'Subject': sub_str})

        weighted_p = weighted_soft_ensemble(test_probs_list, weights)
        results.append(
            {**calculate_metrics(y_test, weighted_p, FIXED_THRESHOLD), 'Model': 'Ensemble (Weighted Baseline)',
             'Subject': sub_str})

        # -------- The Proposed Rigorous Architecture --------
        smoothed_weighted_p = temporal_smooth_probs(weighted_p)
        results.append(
            {**calculate_metrics(y_test, smoothed_weighted_p, opt_thr), 'Model': 'Ensemble (Rigorous Proposed)',
             'Subject': sub_str})

    # Save and summarize
    per_subject_df = pd.DataFrame(results)
    per_subject_df.to_csv(OUTPUT_PER_SUBJECT_CSV, index=False)

    if not per_subject_df.empty:
        metric_cols = ['Decision Threshold', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Balanced Accuracy',
                       'Precision', 'Seizure F1-Score']
        grouped = per_subject_df.groupby('Model', dropna=False)

        mean_df = grouped[metric_cols].mean().reset_index().merge(grouped.size().reset_index(name='N_Subjects'),
                                                                  on='Model')
        mean_df.to_csv(OUTPUT_MACRO_MEAN_CSV, index=False)

        # Log quick summary
        prop_rows = per_subject_df[per_subject_df['Model'] == 'Ensemble (Rigorous Proposed)']
        logging.info('=== FINAL RIGOROUS TEST SET METRICS ===')
        logging.info(f"Macro Sensitivity : {prop_rows['Sensitivity'].mean():.4f}")


if __name__ == '__main__':
    main()