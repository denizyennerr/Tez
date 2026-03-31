import os
import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import tensorflow as tf


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class DatasetInfo:
    data: str
    models_dir: str
    weight: float


# Epoch lengths strictly ordered from 0.5s to 10.0s
DATASETS_INFO: Dict[str, DatasetInfo] = {
    '0.5s': DatasetInfo(
        data='processed_master_datasets/master_dataset_0.5s.npz',
        models_dir='saved_outputs_hybrid/20260324-134941_0.5s_CNN_LSTM_ATTENTION', weight=1.0),
    '1.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_1.0s.npz',
        models_dir='saved_outputs_hybrid/20260325-033656_1.0s_CNN_LSTM_ATTENTION', weight=1.0),
    '2.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_2.0s.npz',
        models_dir='saved_outputs_hybrid/20260325-115542_2.0s_CNN_LSTM_ATTENTION', weight=1.0),
    '4.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_4.0s.npz',
        models_dir='saved_outputs_hybrid/20260325-132117_4.0s_CNN_LSTM_ATTENTION', weight=1.0),
    '5.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_5.0s.npz',
        models_dir='saved_outputs_hybrid/20260325-140449_5.0s_CNN_LSTM_ATTENTION', weight=1.0),
    '10.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_10.0s.npz',
        models_dir='saved_outputs_hybrid/20260325-144433_10.0s_CNN_LSTM_ATTENTION', weight=1.0),
}

REFERENCE_KEY = '0.5s'
BATCH_SIZE = 128
FIXED_THRESHOLD = 0.35
MIN_MODELS_FOR_ENSEMBLE = 2

OUTPUT_DIR = os.path.abspath(os.path.join('saved_outputs_hybrid', 'ensemble_results_final'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_POOLED_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_pooled_metrics.csv')
OUTPUT_MACRO_MEAN_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_macro_mean.csv')
OUTPUT_MACRO_STD_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_macro_std.csv')
OUTPUT_PER_SUBJECT_CSV = os.path.join(OUTPUT_DIR, 'decision_fusion_per_subject.csv')

# =============================================================================
# 2. UTILITIES
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


def window_level_align(coarse_probs: np.ndarray, target_length: int) -> np.ndarray:
    """
    Yeni veri seti yapımız 10 saniyenin tam katı olduğu için
    interpolasyon yerine kusursuz hizalama (np.repeat) yapar.
    """
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
    """Load reference labels (0.5s) to establish subject list and target lengths."""
    ref_path_clean = os.path.abspath(reference_path)
    logging.info('Loading reference dataset: %s', ref_path_clean)
    ref_data = np.load(ref_path_clean)
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
    # THE FIX: Keras/TF backend on Windows fails if it parses backslashes as escape sequences.
    # We must explicitly force forward slashes for the Keras load_model function.
    safe_model_path = model_path.replace('\\', '/')
    model = load_model(safe_model_path, compile=False)

    dataset = tf.data.Dataset.from_tensor_slices(x_subject).batch(BATCH_SIZE)
    probs = model.predict(dataset, verbose=0).ravel()

    del model
    K.clear_session()
    gc.collect()

    return probs


def soft_ensemble(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(probs_list), axis=0)


def hard_vote_ensemble(probs_list: List[np.ndarray], threshold: float = FIXED_THRESHOLD) -> np.ndarray:
    votes = (np.vstack(probs_list) >= threshold).astype(int)
    return np.mean(votes, axis=0)


# =============================================================================
# 3. PREDICTION COLLECTION
# =============================================================================

def collect_probabilities(
        subjects: np.ndarray,
        target_len_by_subject: Dict[str, int],
        datasets_info: Dict[str, DatasetInfo]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate aligned probabilities per resolution and subject."""
    all_probs: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in datasets_info.keys()}

    for res_key, info in datasets_info.items():
        data_path = os.path.abspath(info.data)
        logging.info('Loading dataset for resolution %s: %s', res_key, data_path)

        if not os.path.exists(data_path):
            logging.warning('Missing dataset file: %s', data_path)
            continue

        data = np.load(data_path)
        X = data['X']
        X = np.swapaxes(X, 0, 1)
        s = data['s']

        for subject in subjects:
            sub_str = str(subject)
            base_dir = os.path.abspath(info.models_dir)

            # --- THE FIX: Smartly search for the model using common variations ---
            possible_names = [
                # 1. Looking directly in the base directory
                f'best_model_subject_{sub_str}.keras',
                f'best_model_{sub_str}.keras',
                f'{sub_str}_best_model.keras',
                f'model_{sub_str}.keras',
                f'{sub_str}.keras',
                os.path.join(sub_str, 'best_model.keras'),
                f'best_model_subject_{sub_str}.h5',
                f'best_model_{sub_str}.h5',
                f'{sub_str}_best_model.h5',
                os.path.join(sub_str, 'best_model.h5'),

                # 2. Looking inside the "models" subdirectory
                os.path.join('models', f'best_model_subject_{sub_str}.keras'),
                os.path.join('models', f'best_model_{sub_str}.keras'),
                os.path.join('models', f'{sub_str}_best_model.keras'),
                os.path.join('models', f'model_{sub_str}.keras'),
                os.path.join('models', f'{sub_str}.keras'),
                os.path.join('models', sub_str, 'best_model.keras'),
                os.path.join('models', f'best_model_subject_{sub_str}.h5'),
                os.path.join('models', f'best_model_{sub_str}.h5'),
                os.path.join('models', f'{sub_str}_best_model.h5'),
                os.path.join('models', sub_str, 'best_model.h5')
            ]

            model_path = None
            for name in possible_names:
                p = os.path.join(base_dir, name)
                if os.path.exists(p):
                    model_path = p
                    break

            # If still not found, print out the directory contents to help you debug
            if model_path is None:
                try:
                    dir_contents = os.listdir(base_dir)
                    sample_files = ", ".join(dir_contents[:7]) + ("..." if len(dir_contents) > 7 else "")
                except FileNotFoundError:
                    sample_files = "[Directory itself not found!]"

                logging.warning(
                    f'Missing model for {res_key} subject {sub_str}. '
                    f'Directory: {base_dir} | Sample contents: {sample_files}'
                )
                continue

            mask = s == subject
            if not np.any(mask):
                logging.warning('No samples for %s subject %s', res_key, sub_str)
                continue

            probs_raw = predict_subject_probs(X[mask], model_path)
            aligned = window_level_align(probs_raw, target_len_by_subject[sub_str])
            all_probs[res_key][sub_str] = aligned

        del data, X, s
        gc.collect()

    return all_probs


# =============================================================================
# 4. EVALUATION
# =============================================================================

def build_per_subject_results(
        subjects: np.ndarray,
        y_true_by_subject: Dict[str, np.ndarray],
        probs_by_resolution: Dict[str, Dict[str, np.ndarray]],
        datasets_info: Dict[str, DatasetInfo]
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []

    for subject in subjects:
        sub_str = str(subject)
        y_true = y_true_by_subject[sub_str]

        # Bireysel modeller — kendi fixed threshold ile karşılaştırma için
        for res_key in datasets_info.keys():
            if sub_str not in probs_by_resolution[res_key]:
                continue
            probs = probs_by_resolution[res_key][sub_str]
            metrics = calculate_metrics(y_true, probs, threshold=FIXED_THRESHOLD)
            metrics['Model'] = f'{res_key} Individual'
            metrics['Subject'] = sub_str
            results.append(metrics)

        # Ensemble listesi oluşturma
        probs_list = []
        for res_key, info in datasets_info.items():
            p = probs_by_resolution[res_key].get(sub_str)
            if p is not None:
                probs_list.append(p)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE:
            continue

        # --- Soft Mean (sabit threshold) ---
        soft_probs = soft_ensemble(probs_list)
        soft_metrics = calculate_metrics(y_true, soft_probs, threshold=FIXED_THRESHOLD)
        soft_metrics['Model'] = 'Ensemble (Soft Vote)'
        soft_metrics['Subject'] = sub_str
        results.append(soft_metrics)

        # --- Hard Vote (sabit threshold) ---
        hard_probs = hard_vote_ensemble(probs_list, threshold=FIXED_THRESHOLD)
        hard_metrics = calculate_metrics(y_true, hard_probs, threshold=0.5)
        hard_metrics['Model'] = 'Ensemble (Hard Vote)'
        hard_metrics['Subject'] = sub_str
        results.append(hard_metrics)

    return results


def build_pooled_results(
        subjects: np.ndarray,
        y_true_by_subject: Dict[str, np.ndarray],
        probs_by_resolution: Dict[str, Dict[str, np.ndarray]],
        datasets_info: Dict[str, DatasetInfo]
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []

    def concat_for_model(model_key: str) -> Tuple[np.ndarray, np.ndarray]:
        ys, ps = [], []
        for subject in subjects:
            sub_str = str(subject)
            if sub_str not in probs_by_resolution[model_key]:
                continue
            ys.append(y_true_by_subject[sub_str])
            ps.append(probs_by_resolution[model_key][sub_str])
        if not ys:
            return np.array([]), np.array([])
        return np.concatenate(ys), np.concatenate(ps)

    # Bireysel modeller
    for res_key in datasets_info.keys():
        y_true, probs = concat_for_model(res_key)
        if len(y_true) > 0:
            metrics = calculate_metrics(y_true, probs, threshold=FIXED_THRESHOLD)
            metrics['Model'] = f'{res_key} Individual'
            results.append(metrics)

    # Ensemble'lar
    ys_all = []
    probs_soft_all = []
    probs_hard_all = []

    for subject in subjects:
        sub_str = str(subject)
        y_true = y_true_by_subject[sub_str]

        probs_list = []
        for res_key, info in datasets_info.items():
            p = probs_by_resolution[res_key].get(sub_str)
            if p is not None:
                probs_list.append(p)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE:
            continue

        soft_p = soft_ensemble(probs_list)
        hard_p = hard_vote_ensemble(probs_list, threshold=FIXED_THRESHOLD)

        ys_all.append(y_true)
        probs_soft_all.append(soft_p)
        probs_hard_all.append(hard_p)

    if ys_all:
        y_all = np.concatenate(ys_all)

        soft_m = calculate_metrics(y_all, np.concatenate(probs_soft_all), threshold=FIXED_THRESHOLD)
        soft_m['Model'] = 'Ensemble (Soft Vote)'
        results.append(soft_m)

        hard_m = calculate_metrics(y_all, np.concatenate(probs_hard_all), threshold=0.5)
        hard_m['Model'] = 'Ensemble (Hard Vote)'
        results.append(hard_m)

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
# 5. MAIN
# =============================================================================

def main() -> None:
    if REFERENCE_KEY not in DATASETS_INFO:
        raise KeyError(f'Reference key {REFERENCE_KEY} not found in DATASETS_INFO')

    subjects, y_true_by_subject, target_len_by_subject = load_reference_subjects(
        DATASETS_INFO[REFERENCE_KEY].data
    )

    probs_by_resolution = collect_probabilities(subjects, target_len_by_subject, DATASETS_INFO)

    # -----------------------------------------------------------------------
    # Değerlendirme
    # -----------------------------------------------------------------------
    per_subject_results = build_per_subject_results(
        subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO
    )
    per_subject_df = pd.DataFrame(per_subject_results)
    per_subject_df.to_csv(OUTPUT_PER_SUBJECT_CSV, index=False)

    pooled_results = build_pooled_results(
        subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO
    )
    pooled_df = pd.DataFrame(pooled_results)
    pooled_df.to_csv(OUTPUT_POOLED_CSV, index=False)

    if not per_subject_df.empty:
        macro_mean_df, macro_std_df = build_macro_summary(per_subject_df)
        macro_mean_df.to_csv(OUTPUT_MACRO_MEAN_CSV, index=False)
        macro_std_df.to_csv(OUTPUT_MACRO_STD_CSV, index=False)

    # -----------------------------------------------------------------------
    # Terminal özeti
    # -----------------------------------------------------------------------
    logging.info('=' * 60)
    logging.info('ENSEMBLE FUSION — SONUÇ ÖZETİ')
    logging.info('=' * 60)

    for ensemble_type in ['Ensemble (Soft Vote)', 'Ensemble (Hard Vote)']:
        ensemble_rows = per_subject_df[per_subject_df['Model'] == ensemble_type]
        if not ensemble_rows.empty:
            logging.info(f'--- {ensemble_type.upper()} ---')
            logging.info(
                'Macro AUROC       : %.4f ± %.4f',
                ensemble_rows['AUROC'].mean(), ensemble_rows['AUROC'].std()
            )
            logging.info(
                'Macro Sensitivity : %.4f ± %.4f',
                ensemble_rows['Sensitivity'].mean(), ensemble_rows['Sensitivity'].std()
            )
            logging.info(
                'Macro Specificity : %.4f ± %.4f',
                ensemble_rows['Specificity'].mean(), ensemble_rows['Specificity'].std()
            )
            logging.info(
                'Macro F1-Score    : %.4f ± %.4f',
                ensemble_rows['Seizure F1-Score'].mean(), ensemble_rows['Seizure F1-Score'].std()
            )
            logging.info('')

    logging.info('Saved per-subject metrics : %s', OUTPUT_PER_SUBJECT_CSV)
    logging.info('Saved pooled metrics      : %s', OUTPUT_POOLED_CSV)
    logging.info('Saved macro-mean metrics  : %s', OUTPUT_MACRO_MEAN_CSV)


if __name__ == '__main__':
    main()
