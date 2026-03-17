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
from scipy.stats import rankdata


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class DatasetInfo:
    data: str
    models_dir: str
    weight: float


DATASETS_INFO: Dict[str, DatasetInfo] = {
    '0.5s': DatasetInfo(data='master_dataset_0.5s.npz', models_dir='saved_outputs/20260311-144059_0.5s/models',
                        weight=1.0),
    '1s': DatasetInfo(data='master_dataset_1s.npz', models_dir='saved_outputs/20260304-135510-1s/models', weight=4.0),
    '2s': DatasetInfo(data='master_dataset_2s.npz', models_dir='saved_outputs/20260303-162053-2s/models', weight=1.0),
    '4s': DatasetInfo(data='master_dataset_4s.npz', models_dir='saved_outputs/20260304-091718-4s/models', weight=0.0),
    '5s': DatasetInfo(data='master_dataset_5s.npz', models_dir='saved_outputs/20260309-115134_5s/models', weight=0.0),
    '10s': DatasetInfo(data='master_dataset_10s.npz', models_dir='saved_outputs/20260309-155522_10s/models',
                       weight=0.0),
}

REFERENCE_KEY = '1s'
BATCH_SIZE = 64
FIXED_THRESHOLD = 0.3
MIN_MODELS_FOR_ENSEMBLE = 2

OUTPUT_DIR = os.path.join('saved_outputs', 'ensemble_results_final')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_POOLED_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_pooled_metrics.csv')
OUTPUT_MACRO_MEAN_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_macro_mean.csv')
OUTPUT_MACRO_STD_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_macro_std.csv')
OUTPUT_PER_SUBJECT_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_per_subject.csv')

# =============================================================================
# 2. ALIGNMENT & METRICS
# =============================================================================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def time_based_align(coarse_probs: np.ndarray, target_length: int, coarse_res: float,
                     ref_res: float = 1.0) -> np.ndarray:
    """Note: This requires EDF-file level alignment to be perfectly accurate."""
    if len(coarse_probs) == 0:
        return np.zeros(target_length, dtype=coarse_probs.dtype)
    t_starts = np.arange(target_length) * ref_res
    coarse_indices = np.floor(t_starts / coarse_res).astype(int)
    coarse_indices = np.clip(coarse_indices, 0, len(coarse_probs) - 1)
    return coarse_probs[coarse_indices]


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = None) -> Dict[str, float]:
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

        # Rigorous threshold search if dynamic
        if threshold is None:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = np.divide(2 * (precisions * recalls), (precisions + recalls),
                                  out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
            best_idx = np.argmax(f1_scores)
            threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    else:
        auroc, auprc = np.nan, np.nan
        if threshold is None: threshold = FIXED_THRESHOLD

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        'Decision Threshold': round(float(threshold), 4),
        'AUROC': round(float(auroc), 4),
        'AUPRC': round(float(auprc), 4),
        'Sensitivity': round(float(sensitivity), 4),
        'Specificity': round(float(specificity), 4),
        'Balanced Accuracy': round(float(balanced_acc), 4),
        'Seizure F1-Score': round(float(f1_score), 4),
    }


# =============================================================================
# 3. ADVANCED ENSEMBLE STRATEGIES
# =============================================================================

def soft_ensemble(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(probs_list), axis=0)


def weighted_soft_ensemble(probs_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    return np.average(np.vstack(probs_list), axis=0, weights=np.asarray(weights, dtype=float))


def hard_vote_ensemble(probs_list: List[np.ndarray]) -> np.ndarray:
    votes = (np.vstack(probs_list) >= FIXED_THRESHOLD).astype(int)
    return np.mean(votes, axis=0)


def rank_ensemble(probs_list: List[np.ndarray]) -> np.ndarray:
    """Rigorous rank-based fusion (robust to miscalibrated probabilities)."""
    ranks = [rankdata(p) / len(p) for p in probs_list]
    return np.mean(np.vstack(ranks), axis=0)


def dynamic_confidence_ensemble(probs_list: List[np.ndarray], gamma: float = 2.0) -> np.ndarray:
    """
    State-of-the-Art Confidence Fusion:
    Weights models dynamically per-window based on their distance from 0.5 (uncertainty).
    """
    probs_arr = np.vstack(probs_list)
    confidence = np.abs(probs_arr - 0.5) ** gamma
    confidence = np.clip(confidence, 1e-6, None)  # Prevent zero division

    weights = confidence / np.sum(confidence, axis=0)
    return np.sum(probs_arr * weights, axis=0)


# =============================================================================
# 4. PREDICTION COLLECTION & EVALUATION
# =============================================================================
def load_reference_subjects(reference_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
    ref_data = np.load(reference_path)
    subjects = np.unique(ref_data['s'])
    y_true_by_subject, target_len_by_subject = {}, {}
    for subject in subjects:
        mask = ref_data['s'] == subject
        y_sub = ref_data['y'][mask]
        sub_str = str(subject)
        y_true_by_subject[sub_str] = y_sub
        target_len_by_subject[sub_str] = len(y_sub)
    del ref_data;
    gc.collect()
    return subjects, y_true_by_subject, target_len_by_subject


def predict_subject_probs(x_subject: np.ndarray, model_path: str) -> np.ndarray:
    model = load_model(model_path)
    probs = model.predict(x_subject, batch_size=BATCH_SIZE, verbose=0).ravel()
    del model;
    K.clear_session()
    return probs


def collect_probabilities(subjects: np.ndarray, target_len_by_subject: Dict[str, int],
                          datasets_info: Dict[str, DatasetInfo]) -> Dict[str, Dict[str, np.ndarray]]:
    all_probs = {k: {} for k in datasets_info.keys()}
    for res_key, info in datasets_info.items():
        if not os.path.exists(info.data): continue
        coarse_res_float = float(res_key.replace('s', ''))
        data = np.load(info.data)
        X, s = data['X'], data['s']

        for subject in subjects:
            sub_str = str(subject)
            model_path = os.path.join(info.models_dir, f'best_model_subject_{sub_str}.keras')
            if not os.path.exists(model_path): continue
            mask = s == subject
            if not np.any(mask): continue

            probs_raw = predict_subject_probs(X[mask], model_path)
            all_probs[res_key][sub_str] = time_based_align(probs_raw, target_len_by_subject[sub_str], coarse_res_float,
                                                           1.0)

        del data, X, s;
        gc.collect()
    return all_probs


def build_per_subject_results(subjects: np.ndarray, y_true_by_subject: Dict[str, np.ndarray],
                              probs_by_resolution: Dict[str, Dict[str, np.ndarray]],
                              datasets_info: Dict[str, DatasetInfo]) -> List[Dict[str, float]]:
    results = []
    for subject in subjects:
        sub_str = str(subject)
        y_true = y_true_by_subject[sub_str]

        for res_key in datasets_info.keys():
            if sub_str in probs_by_resolution[res_key]:
                metrics = calculate_metrics(y_true, probs_by_resolution[res_key][sub_str], threshold=FIXED_THRESHOLD)
                metrics['Model'] = f'{res_key} Individual'
                metrics['Subject'] = sub_str
                results.append(metrics)

        probs_list, weights_list = [], []
        for res_key, info in datasets_info.items():
            if info.weight <= 0.0: continue  # Strict Exclusion of bad models
            p = probs_by_resolution[res_key].get(sub_str)
            if p is not None:
                probs_list.append(p)
                weights_list.append(info.weight)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE: continue

        for name, probs in [
            ('Ensemble (Soft Mean)', soft_ensemble(probs_list)),
            ('Ensemble (Weighted Soft)', weighted_soft_ensemble(probs_list, weights_list)),
            ('Ensemble (Hard Vote)', hard_vote_ensemble(probs_list)),
            ('Ensemble (Rank Average)', rank_ensemble(probs_list)),
            ('Ensemble (Dynamic Confidence)', dynamic_confidence_ensemble(probs_list))
        ]:
            metrics = calculate_metrics(y_true, probs, threshold=None)
            metrics['Model'] = name
            metrics['Subject'] = sub_str
            results.append(metrics)

    return results


def build_pooled_results(subjects: np.ndarray, y_true_by_subject: Dict[str, np.ndarray],
                         probs_by_resolution: Dict[str, Dict[str, np.ndarray]],
                         datasets_info: Dict[str, DatasetInfo]) -> List[Dict[str, float]]:
    results = []

    def concat_for_model(model_key: str) -> Tuple[np.ndarray, np.ndarray]:
        ys, ps = [], []
        for sub_str in map(str, subjects):
            if sub_str in probs_by_resolution[model_key]:
                ys.append(y_true_by_subject[sub_str])
                ps.append(probs_by_resolution[model_key][sub_str])
        return (np.concatenate(ys), np.concatenate(ps)) if ys else (np.array([]), np.array([]))

    for res_key in datasets_info.keys():
        y_true, probs = concat_for_model(res_key)
        if len(y_true) > 0:
            metrics = calculate_metrics(y_true, probs, threshold=FIXED_THRESHOLD)
            metrics['Model'] = f'{res_key} Individual'
            results.append(metrics)

    ys_all, probs_soft_all, probs_weighted_all, probs_hard_all, probs_rank_all, probs_dyn_all = [], [], [], [], [], []
    for sub_str in map(str, subjects):
        y_true = y_true_by_subject[sub_str]

        probs_list, weights_list = [], []
        for res_key, info in datasets_info.items():
            if info.weight <= 0.0: continue
            p = probs_by_resolution[res_key].get(sub_str)
            if p is not None:
                probs_list.append(p)
                weights_list.append(info.weight)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE: continue

        ys_all.append(y_true)
        probs_soft_all.append(soft_ensemble(probs_list))
        probs_weighted_all.append(weighted_soft_ensemble(probs_list, weights_list))
        probs_hard_all.append(hard_vote_ensemble(probs_list))
        probs_rank_all.append(rank_ensemble(probs_list))
        probs_dyn_all.append(dynamic_confidence_ensemble(probs_list))

    if ys_all:
        y_all = np.concatenate(ys_all)
        for name, probs in [
            ('Ensemble (Soft Mean)', np.concatenate(probs_soft_all)),
            ('Ensemble (Weighted Soft)', np.concatenate(probs_weighted_all)),
            ('Ensemble (Hard Vote)', np.concatenate(probs_hard_all)),
            ('Ensemble (Rank Average)', np.concatenate(probs_rank_all)),
            ('Ensemble (Dynamic Confidence)', np.concatenate(probs_dyn_all))
        ]:
            metrics = calculate_metrics(y_all, probs, threshold=None)
            metrics['Model'] = name
            results.append(metrics)

    return results


def build_macro_summary(per_subject_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ['Decision Threshold', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Balanced Accuracy',
                   'Seizure F1-Score']
    grouped = per_subject_df.groupby('Model', dropna=False)
    mean_df, std_df = grouped[metric_cols].mean().reset_index(), grouped[metric_cols].std().reset_index()
    counts = grouped.size().reset_index(name='N_Subjects')

    cols = ['Model', 'N_Subjects'] + metric_cols
    return mean_df.merge(counts, on='Model')[cols], std_df.merge(counts, on='Model')[cols]


# =============================================================================
# 5. MAIN
# =============================================================================
def main() -> None:
    if REFERENCE_KEY not in DATASETS_INFO: raise KeyError(f'Reference key {REFERENCE_KEY} not found')
    subjects, y_true_by_subject, target_len_by_subject = load_reference_subjects(DATASETS_INFO[REFERENCE_KEY].data)
    probs_by_resolution = collect_probabilities(subjects, target_len_by_subject, DATASETS_INFO)

    per_subject_df = pd.DataFrame(
        build_per_subject_results(subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO))
    per_subject_df.to_csv(OUTPUT_PER_SUBJECT_CSV, index=False)

    pooled_df = pd.DataFrame(build_pooled_results(subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO))
    pooled_df.to_csv(OUTPUT_POOLED_CSV, index=False)

    if not per_subject_df.empty:
        macro_mean_df, macro_std_df = build_macro_summary(per_subject_df)
        macro_mean_df.to_csv(OUTPUT_MACRO_MEAN_CSV, index=False)
        macro_std_df.to_csv(OUTPUT_MACRO_STD_CSV, index=False)

    logging.info('Saved final metrics successfully.')


if __name__ == '__main__':
    main()