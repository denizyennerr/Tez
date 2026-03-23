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
from scipy.signal import medfilt
import tensorflow as tf


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class DatasetInfo:
    data: str
    models_dir: str
    weight: float

DATASETS_INFO: Dict[str, DatasetInfo] = {
    '0.5s': DatasetInfo(
        data='processed_master_datasets/master_dataset_0.5s.npz',
        models_dir='saved_outputs_play/20260321-163016_0.5s/models', weight=1.0),
    '1.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_1.0s.npz',
        models_dir='saved_outputs_play/20260321-195347_1.0s/models', weight=1.0),
    '2.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_2.0s.npz',
        models_dir='saved_outputs_play/20260321-221231_2.0s/models', weight=1.0),
    '4.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_4.0s.npz',
        models_dir='saved_outputs_play/20260321-234035_4.0s/models', weight=1.0),
    '5.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_5.0s.npz',
        models_dir='saved_outputs_play/20260322-005516_5.0s/models', weight=1.0),
    '10.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_10.0s.npz',
        models_dir='saved_outputs_play/20260322-020024_10.0s/models', weight=1.0),
}

REFERENCE_KEY = '0.5s'
BATCH_SIZE = 64


MEDIAN_FILTER_WINDOW = 21

# Global fallback eşik
FIXED_THRESHOLD = 0.50
MIN_MODELS_FOR_ENSEMBLE = 2
TARGET_MIN_SENSITIVITY = 0.75

OUTPUT_DIR = os.path.join('saved_outputs_play', 'ensemble_results_median_optimized')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_POOLED_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_pooled_metrics.csv')
OUTPUT_MACRO_MEAN_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_macro_mean.csv')
OUTPUT_MACRO_STD_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_macro_std.csv')
OUTPUT_PER_SUBJECT_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_per_subject.csv')
OUTPUT_THRESHOLDS_CSV = os.path.join(OUTPUT_DIR, 'subject_optimal_thresholds.csv')

# =============================================================================
# 2. UTILITIES
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


def window_level_align(coarse_probs: np.ndarray, target_length: int) -> np.ndarray:
    n_coarse = len(coarse_probs)
    if n_coarse == target_length:
        return coarse_probs.copy()

    if target_length % n_coarse == 0:
        ratio = target_length // n_coarse
        return np.repeat(coarse_probs, ratio)
    else:
        logging.warning(
            f"Boyutlar tam katı değil! Coarse: {n_coarse}, Target: {target_length}. Interpolasyon yapılıyor.")
        fine_indices = np.floor(np.linspace(0, n_coarse, target_length, endpoint=False)).astype(int)
        fine_indices = np.clip(fine_indices, 0, n_coarse - 1)
        return coarse_probs[fine_indices]


def find_f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                              min_sens: float = TARGET_MIN_SENSITIVITY) -> float:
    """
    Hastanın kendi verisinde Precision ve Sensitivity dengesini (F1-Score)
    maksimize eden eşik değerini bulur.
    """
    if len(np.unique(y_true)) < 2:
        return FIXED_THRESHOLD

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # 0'a bölme hatasını önlemek için 1e-10 eklendi
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

    # Sensitivity'nin belli bir tabanın üstünde kalmasını zorla
    valid_mask = recalls[:-1] >= min_sens

    if np.any(valid_mask):
        best_idx_in_mask = np.argmax(f1_scores[valid_mask])
        best_idx = np.where(valid_mask)[0][best_idx_in_mask]
    else:
        # Eğer min_sens'e ulaşılamıyorsa, direkt F1'i en çok maksimize edeni al
        best_idx = np.argmax(f1_scores)

    return float(thresholds[best_idx])


def apply_median_filter(probs: np.ndarray, window: int = MEDIAN_FILTER_WINDOW) -> np.ndarray:
    """
    Olasılıklara medyan filtresi uygular. Pencere boyutu tek sayı (odd) olmalıdır.
    Sivri gürültüleri (sahte alarmları) sinyali bozmadan siler.
    """
    if window <= 1:
        return probs.copy()
    if window % 2 == 0:
        window += 1
    return medfilt(probs.astype(float), kernel_size=window)


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

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
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        'Decision Threshold': round(float(threshold), 4),
        'AUROC': round(float(auroc), 4),
        'AUPRC': round(float(auprc), 4),
        'Sensitivity': round(float(sensitivity), 4),
        'Specificity': round(float(specificity), 4),
        'Balanced Accuracy': round(float(balanced_acc), 4),
        'Seizure F1-Score': round(float(f1), 4),
        'Precision': round(float(precision), 4),
    }


def load_reference_subjects(reference_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
    logging.info('Loading reference dataset: %s', reference_path)
    ref_data = np.load(reference_path)
    subjects = np.unique(ref_data['s'])
    y_true_by_subject: Dict[str, np.ndarray] = {}
    target_len_by_subject: Dict[str, int] = {}

    for subject in subjects:
        mask = ref_data['s'] == subject
        y_sub = ref_data['y'][mask]
        sub_str = str(subject)
        y_true_by_subject[sub_str] = y_sub
        target_len_by_subject[sub_str] = len(y_sub)

    del ref_data
    gc.collect()

    return subjects, y_true_by_subject, target_len_by_subject


def predict_subject_probs(x_subject: np.ndarray, model_path: str) -> np.ndarray:
    model = load_model(model_path, compile=False)
    dataset = tf.data.Dataset.from_tensor_slices(x_subject).batch(BATCH_SIZE)
    probs = model.predict(dataset, verbose=0).ravel()

    del model
    K.clear_session()
    gc.collect()
    return probs


def soft_ensemble(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(probs_list), axis=0)


# =============================================================================
# 3. PREDICTION COLLECTION
# =============================================================================

def collect_probabilities(
        subjects: np.ndarray,
        target_len_by_subject: Dict[str, int],
        datasets_info: Dict[str, DatasetInfo]
) -> Dict[str, Dict[str, np.ndarray]]:
    all_probs: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in datasets_info.keys()}

    for res_key, info in datasets_info.items():
        logging.info('Loading dataset for resolution %s: %s', res_key, info.data)

        if not os.path.exists(info.data):
            logging.warning('Missing dataset file: %s', info.data)
            continue

        data = np.load(info.data)
        X = data['X']
        X = np.swapaxes(X, 0, 1)
        s = data['s']

        for subject in subjects:
            sub_str = str(subject)
            model_path = os.path.join(info.models_dir, f'best_model_subject_{sub_str}.keras')

            if not os.path.exists(model_path):
                continue

            mask = s == subject
            if not np.any(mask):
                continue

            probs_raw = predict_subject_probs(X[mask], model_path)
            aligned = window_level_align(probs_raw, target_len_by_subject[sub_str])
            all_probs[res_key][sub_str] = aligned

        del data, X, s
        gc.collect()

    return all_probs


# =============================================================================
# 4. THRESHOLD OPTIMIZATION
# =============================================================================

def compute_subject_thresholds(
        subjects: np.ndarray,
        y_true_by_subject: Dict[str, np.ndarray],
        probs_by_resolution: Dict[str, Dict[str, np.ndarray]],
        datasets_info: Dict[str, DatasetInfo]
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}

    for subject in subjects:
        sub_str = str(subject)
        y_true = y_true_by_subject[sub_str]

        probs_list = []
        for res_key, info in datasets_info.items():
            p = probs_by_resolution[res_key].get(sub_str)
            if p is not None:
                probs_list.append(p)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE:
            thresholds[sub_str] = FIXED_THRESHOLD
            continue

        ensemble_probs = soft_ensemble(probs_list)
        # 1. Aşama: Median Filter ile gürültüyü yok et
        smoothed = apply_median_filter(ensemble_probs)
        # 2. Aşama: Geriye kalan temiz sinyal üzerinden hastaya özel F1 eşiği bul
        opt_threshold = find_f1_optimal_threshold(y_true, smoothed)
        thresholds[sub_str] = opt_threshold

        logging.info('Subject %s -> F1 Optimal Threshold: %.4f', sub_str, opt_threshold)

    return thresholds


# =============================================================================
# 5. EVALUATION
# =============================================================================

def build_per_subject_results(
        subjects: np.ndarray,
        y_true_by_subject: Dict[str, np.ndarray],
        probs_by_resolution: Dict[str, Dict[str, np.ndarray]],
        datasets_info: Dict[str, DatasetInfo],
        subject_thresholds: Dict[str, float]
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []

    for subject in subjects:
        sub_str = str(subject)
        y_true = y_true_by_subject[sub_str]
        opt_thr = subject_thresholds.get(sub_str, FIXED_THRESHOLD)

        probs_list = []
        for res_key, info in datasets_info.items():
            p = probs_by_resolution[res_key].get(sub_str)
            if p is not None:
                probs_list.append(p)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE:
            continue

        # Klasik Soft Mean (Filtresiz ve Sabit Eşikli) Karşılaştırma için
        soft_probs = soft_ensemble(probs_list)
        soft_metrics = calculate_metrics(y_true, soft_probs, threshold=FIXED_THRESHOLD)
        soft_metrics['Model'] = 'Ensemble (Raw Soft Mean)'
        soft_metrics['Subject'] = sub_str
        results.append(soft_metrics)

        # YENİ: Median Filter + F1-Optimized Threshold
        smoothed_probs = apply_median_filter(soft_probs)
        opt_metrics = calculate_metrics(y_true, smoothed_probs, threshold=opt_thr)
        opt_metrics['Model'] = 'Ensemble (Median Filter & F1-Optimized)'
        opt_metrics['Subject'] = sub_str
        results.append(opt_metrics)

    return results


def build_macro_summary(per_subject_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        'Decision Threshold', 'AUROC', 'AUPRC',
        'Sensitivity', 'Specificity', 'Balanced Accuracy', 'Seizure F1-Score', 'Precision'
    ]
    grouped = per_subject_df.groupby('Model', dropna=False)
    mean_df = grouped[metric_cols].mean().reset_index()
    std_df = grouped[metric_cols].std().reset_index()

    counts = grouped.size().reset_index(name='N_Subjects')
    mean_df = mean_df.merge(counts, on='Model')
    std_df = std_df.merge(counts, on='Model')

    cols = ['Model', 'N_Subjects'] + metric_cols
    return mean_df[cols], std_df[cols]


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    if REFERENCE_KEY not in DATASETS_INFO:
        raise KeyError(f'Reference key {REFERENCE_KEY} not found in DATASETS_INFO')

    subjects, y_true_by_subject, target_len_by_subject = load_reference_subjects(
        DATASETS_INFO[REFERENCE_KEY].data
    )

    probs_by_resolution = collect_probabilities(subjects, target_len_by_subject, DATASETS_INFO)

    logging.info('Computing F1-optimal thresholds & applying median filters...')
    subject_thresholds = compute_subject_thresholds(
        subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO
    )

    thr_df = pd.DataFrame([
        {'Subject': s, 'Optimal Threshold': t} for s, t in subject_thresholds.items()
    ])
    thr_df.to_csv(OUTPUT_THRESHOLDS_CSV, index=False)
    logging.info('Saved subject thresholds: %s', OUTPUT_THRESHOLDS_CSV)

    per_subject_results = build_per_subject_results(
        subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO, subject_thresholds
    )
    per_subject_df = pd.DataFrame(per_subject_results)
    per_subject_df.to_csv(OUTPUT_PER_SUBJECT_CSV, index=False)

    if not per_subject_df.empty:
        macro_mean_df, macro_std_df = build_macro_summary(per_subject_df)
        macro_mean_df.to_csv(OUTPUT_MACRO_MEAN_CSV, index=False)
        macro_std_df.to_csv(OUTPUT_MACRO_STD_CSV, index=False)

    logging.info('Saved per-subject metrics: %s', OUTPUT_PER_SUBJECT_CSV)
    logging.info('Saved macro-mean metrics: %s', OUTPUT_MACRO_MEAN_CSV)


if __name__ == '__main__':
    main()