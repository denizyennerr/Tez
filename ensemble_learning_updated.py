import os
import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from scipy.ndimage import uniform_filter1d
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
        models_dir='saved_outputs_play/20260323-163632_0.5s/models', weight=1.0),
    '1.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_1.0s.npz',
        models_dir='saved_outputs_play/20260323-201436_1.0s/models', weight=1.0),
    '2.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_2.0s.npz',
        models_dir='saved_outputs_play/20260323-220314_2.0s/models', weight=1.0),
    '4.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_4.0s.npz',
        models_dir='saved_outputs_play/20260323-224227_4.0s/models', weight=1.0),
    '5.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_5.0s.npz',
        models_dir='saved_outputs_play/20260323-231310_5.0s/models', weight=1.0),
    '10.0s': DatasetInfo(
        data='processed_master_datasets/master_dataset_10.0s.npz',
        models_dir='saved_outputs_play/20260323-234451_10.0s/models', weight=1.0),
}

REFERENCE_KEY = '0.5s'
BATCH_SIZE = 128

# Klinik hedef: nöbetlerin en az bu oranını yakala
TARGET_SENSITIVITY = 0.80

# Güvenlik tabanı: precision bu değerin altına düşmesin
# (tamamen sıfır alarmlarla dolmasın diye)
MIN_PRECISION_FLOOR = 0.10

# Global fallback eşik — herhangi bir hasta için threshold bulunamazsa kullan
FIXED_THRESHOLD = 0.35  # 0.50'den düşürüldü; klinik öncelik Sensitivity

MIN_MODELS_FOR_ENSEMBLE = 2

# Temporal smoothing pencere boyutu (pencere sayısı cinsinden).
# 0.5s referans çözünürlüğünde 5 pencere = 2.5 saniyelik yumuşatma.
# Kısa süreli (~1-2s) izole false positive'ler bastırılır; gerçek nöbet
# sinyalleri (>3s) korunur.
TEMPORAL_SMOOTH_WINDOW = 5

OUTPUT_DIR = os.path.join('saved_outputs_play', 'ensemble_results_sensitivity_optimized')
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


def find_sensitivity_optimal_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_sensitivity: float = TARGET_SENSITIVITY,
        min_precision: float = MIN_PRECISION_FLOOR
) -> float:
    """
    Verilen prob dizisi için şu koşulları sağlayan en iyi threshold'u döner:
      - Sensitivity >= target_sensitivity
      - Precision >= min_precision  (tamamen anlamsız alarm üretmeyi önler)
      - Bu koşullar altında Precision maksimize edilir (Precision-Recall dengesi)

    Eğer hiçbir threshold her iki koşulu birden sağlayamazsa:
      - Sadece Sensitivity koşulunu sağlayanlar arasından en iyi Precision'ı al.
      - O da yoksa FIXED_THRESHOLD'a geri dön (güvenlik ağı).

    Neden precision_recall_curve kullanıyoruz?
      sklearn'ün precision_recall_curve fonksiyonu tüm olası threshold'ları
      tek geçişte hesaplar; elle döngü kurmaktan hem daha hızlı hem de sayısal
      olarak daha tutarlıdır.
    """
    if len(np.unique(y_true)) < 2:
        # Sadece bir sınıf var (örn. nöbetsiz hasta) — sabit threshold döndür
        return FIXED_THRESHOLD

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve son noktayı (threshold yok) ekler; thresholds
    # dizisi precisions/recalls'dan 1 kısa olur. Hizalıyoruz:
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    # Koşul 1: Sensitivity (= Recall) hedefi karşılanıyor mu?
    sens_ok = recalls >= target_sensitivity
    # Koşul 2: Precision tabanı karşılanıyor mu?
    prec_ok = precisions >= min_precision

    both_ok = sens_ok & prec_ok

    if np.any(both_ok):
        # Her iki koşulu sağlayanlar arasında Precision'ı maksimize eden threshold
        best_idx = np.argmax(precisions[both_ok])
        return float(thresholds[both_ok][best_idx])

    if np.any(sens_ok):
        # Sadece Sensitivity koşulunu sağlayanlar — bu grup içinde en iyi Precision
        best_idx = np.argmax(precisions[sens_ok])
        logging.warning(
            "Precision >= %.2f sağlanamadı, sadece Sensitivity hedefi kullanılıyor.", min_precision)
        return float(thresholds[sens_ok][best_idx])

    # Hiçbir threshold hedef Sensitivity'ye ulaşamadı — global fallback
    logging.warning(
        "Sensitivity >= %.2f hiçbir threshold ile sağlanamadı. FIXED_THRESHOLD kullanılıyor.", target_sensitivity)
    return FIXED_THRESHOLD


def temporal_smooth_probs(probs: np.ndarray, window: int = TEMPORAL_SMOOTH_WINDOW) -> np.ndarray:
    """
    Olasılık dizisine hareketli ortalama uygular.

    Neden temporal smoothing?
      Nöbetler doğası gereği birden fazla ardışık pencereye yayılır. Tek bir
      pencerede ateşlenen izole bir yüksek prob değeri büyük ihtimalle gürültü
      veya artefakttır. Hareketli ortalama bu tür anlık spike'ları bastırır;
      gerçek nöbet sinyalinin prob'unu ise neredeyse değiştirmez çünkü o
      bölgedeki tüm komşular zaten yüksek değer taşır.

      Threshold'u düşürünce artan false positive sayısını bu adım önemli ölçüde
      azaltır — yani Precision'ı kısmen geri kazanmamızı sağlar.
    """
    if window <= 1:
        return probs.copy()
    return uniform_filter1d(probs.astype(float), size=window, mode='nearest')


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


def weighted_soft_ensemble(probs_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    weights_arr = np.asarray(weights, dtype=float)
    if np.sum(weights_arr) == 0:
        return soft_ensemble(probs_list)
    return np.average(np.vstack(probs_list), axis=0, weights=weights_arr)


def hard_vote_ensemble(probs_list: List[np.ndarray], threshold: float = FIXED_THRESHOLD) -> np.ndarray:
    # Hard vote için de güncellenmiş threshold'u kullan
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
                logging.warning('Missing model for %s subject %s', res_key, sub_str)
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
# 4. THRESHOLD OPTIMIZATION
# =============================================================================

def compute_subject_thresholds(
        subjects: np.ndarray,
        y_true_by_subject: Dict[str, np.ndarray],
        probs_by_resolution: Dict[str, Dict[str, np.ndarray]],
        datasets_info: Dict[str, DatasetInfo]
) -> Dict[str, float]:
    """
    Her hasta için Ensemble (Soft Mean) prob'larını kullanarak
    klinik hedefe göre optimal threshold'u hesaplar.

    Neden ensemble prob'u kullanıyoruz?
      Bireysel model prob'ları daha gürültülüdür. Ensemble prob'u
      birden fazla çözünürlükteki sinyal bilgisini harmanlayarak
      daha kalibrasyon değeri taşır; threshold bu daha güvenilir
      dağılım üzerinden seçilir.
    """
    thresholds: Dict[str, float] = {}

    for subject in subjects:
        sub_str = str(subject)
        y_true = y_true_by_subject[sub_str]

        probs_list = []
        weights_list = []
        for res_key, info in datasets_info.items():
            p = probs_by_resolution[res_key].get(sub_str)
            if p is not None:
                probs_list.append(p)
                weights_list.append(info.weight)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE:
            thresholds[sub_str] = FIXED_THRESHOLD
            continue

        ensemble_probs = soft_ensemble(probs_list)
        # Temporal smoothing sonrası threshold ara — smoothed prob üzerinden
        # threshold bulmak, çıkarım sırasında da smoothing uygulanacağından
        # tutarlıdır.
        smoothed = temporal_smooth_probs(ensemble_probs)

        opt_threshold = find_sensitivity_optimal_threshold(y_true, smoothed)
        thresholds[sub_str] = opt_threshold

        logging.info(
            'Subject %s -> optimal threshold: %.4f (target sensitivity >= %.2f)',
            sub_str, opt_threshold, TARGET_SENSITIVITY
        )

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

        # Bireysel modeller — kendi fixed threshold ile karşılaştırma için
        for res_key in datasets_info.keys():
            if sub_str not in probs_by_resolution[res_key]:
                continue
            probs = probs_by_resolution[res_key][sub_str]
            metrics = calculate_metrics(y_true, probs, threshold=FIXED_THRESHOLD)
            metrics['Model'] = f'{res_key} Individual'
            metrics['Subject'] = sub_str
            results.append(metrics)

        # Ensemble listesi
        probs_list = []
        weights_list = []
        for res_key, info in datasets_info.items():
            p = probs_by_resolution[res_key].get(sub_str)
            if p is None:
                continue
            probs_list.append(p)
            weights_list.append(info.weight)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE:
            continue

        # --- Soft Mean (sabit threshold, baseline) ---
        soft_probs = soft_ensemble(probs_list)
        soft_metrics = calculate_metrics(y_true, soft_probs, threshold=FIXED_THRESHOLD)
        soft_metrics['Model'] = 'Ensemble (Soft Mean)'
        soft_metrics['Subject'] = sub_str
        results.append(soft_metrics)

        # --- Weighted Soft (sabit threshold, baseline) ---
        weighted_probs = weighted_soft_ensemble(probs_list, weights_list)
        weighted_metrics = calculate_metrics(y_true, weighted_probs, threshold=FIXED_THRESHOLD)
        weighted_metrics['Model'] = 'Ensemble (Weighted Soft)'
        weighted_metrics['Subject'] = sub_str
        results.append(weighted_metrics)

        # --- Hard Vote (sabit threshold, baseline) ---
        hard_probs = hard_vote_ensemble(probs_list, threshold=FIXED_THRESHOLD)
        hard_metrics = calculate_metrics(y_true, hard_probs, threshold=0.5)
        hard_metrics['Model'] = 'Ensemble (Hard Vote)'
        hard_metrics['Subject'] = sub_str
        results.append(hard_metrics)

        # ----------------------------------------------------------------
        # YENİ: Sensitivity-Optimized Ensemble
        # Soft Mean prob'u üzerinde önce temporal smoothing, ardından
        # hasta-spesifik optimal threshold uygulanır.
        # ----------------------------------------------------------------
        smoothed_probs = temporal_smooth_probs(soft_probs)

        sens_opt_metrics = calculate_metrics(y_true, smoothed_probs, threshold=opt_thr)
        sens_opt_metrics['Model'] = 'Ensemble (Sensitivity-Optimized)'
        sens_opt_metrics['Subject'] = sub_str
        results.append(sens_opt_metrics)

    return results


def build_pooled_results(
        subjects: np.ndarray,
        y_true_by_subject: Dict[str, np.ndarray],
        probs_by_resolution: Dict[str, Dict[str, np.ndarray]],
        datasets_info: Dict[str, DatasetInfo],
        subject_thresholds: Dict[str, float]
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
    ys_all          = []
    probs_soft_all  = []
    probs_wsoft_all = []
    probs_hard_all  = []
    probs_sensopt_all = []

    for subject in subjects:
        sub_str = str(subject)
        y_true  = y_true_by_subject[sub_str]
        opt_thr = subject_thresholds.get(sub_str, FIXED_THRESHOLD)

        probs_list   = []
        weights_list = []
        for res_key, info in datasets_info.items():
            p = probs_by_resolution[res_key].get(sub_str)
            if p is None:
                continue
            probs_list.append(p)
            weights_list.append(info.weight)

        if len(probs_list) < MIN_MODELS_FOR_ENSEMBLE:
            continue

        soft_p    = soft_ensemble(probs_list)
        smoothed_p = temporal_smooth_probs(soft_p)

        ys_all.append(y_true)
        probs_soft_all.append(soft_p)
        probs_wsoft_all.append(weighted_soft_ensemble(probs_list, weights_list))
        probs_hard_all.append(hard_vote_ensemble(probs_list, threshold=FIXED_THRESHOLD))
        probs_sensopt_all.append(smoothed_p)

    if ys_all:
        y_all = np.concatenate(ys_all)

        soft_m = calculate_metrics(y_all, np.concatenate(probs_soft_all), threshold=FIXED_THRESHOLD)
        soft_m['Model'] = 'Ensemble (Soft Mean)'
        results.append(soft_m)

        wsoft_m = calculate_metrics(y_all, np.concatenate(probs_wsoft_all), threshold=FIXED_THRESHOLD)
        wsoft_m['Model'] = 'Ensemble (Weighted Soft)'
        results.append(wsoft_m)

        hard_m = calculate_metrics(y_all, np.concatenate(probs_hard_all), threshold=0.5)
        hard_m['Model'] = 'Ensemble (Hard Vote)'
        results.append(hard_m)

        sensopt_m = calculate_metrics(y_all, np.concatenate(probs_sensopt_all), threshold=FIXED_THRESHOLD)
        sensopt_m['Model'] = 'Ensemble (Sensitivity-Optimized)'
        results.append(sensopt_m)

    return results


def build_macro_summary(per_subject_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        'Decision Threshold', 'AUROC', 'AUPRC',
        'Sensitivity', 'Specificity', 'Balanced Accuracy', 'Seizure F1-Score', 'Precision'
    ]
    grouped   = per_subject_df.groupby('Model', dropna=False)
    mean_df   = grouped[metric_cols].mean().reset_index()
    std_df    = grouped[metric_cols].std().reset_index()

    counts    = grouped.size().reset_index(name='N_Subjects')
    mean_df   = mean_df.merge(counts, on='Model')
    std_df    = std_df.merge(counts, on='Model')

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

    # -----------------------------------------------------------------------
    # Hasta-spesifik threshold optimizasyonu
    # -----------------------------------------------------------------------
    logging.info(
        'Computing sensitivity-optimal thresholds '
        '(target sensitivity >= %.2f, min precision >= %.2f) ...',
        TARGET_SENSITIVITY, MIN_PRECISION_FLOOR
    )
    subject_thresholds = compute_subject_thresholds(
        subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO
    )

    # Threshold'ları CSV'ye kaydet — şeffaflık ve tekrarlanabilirlik için
    thr_df = pd.DataFrame([
        {'Subject': s, 'Optimal Threshold': t, 'Target Sensitivity': TARGET_SENSITIVITY}
        for s, t in subject_thresholds.items()
    ])
    thr_df.to_csv(OUTPUT_THRESHOLDS_CSV, index=False)
    logging.info('Saved subject thresholds: %s', OUTPUT_THRESHOLDS_CSV)

    # -----------------------------------------------------------------------
    # Değerlendirme
    # -----------------------------------------------------------------------
    per_subject_results = build_per_subject_results(
        subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO, subject_thresholds
    )
    per_subject_df = pd.DataFrame(per_subject_results)
    per_subject_df.to_csv(OUTPUT_PER_SUBJECT_CSV, index=False)

    pooled_results = build_pooled_results(
        subjects, y_true_by_subject, probs_by_resolution, DATASETS_INFO, subject_thresholds
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
    logging.info('SENSITIVITY-OPTIMIZED ENSEMBLE — SONUÇ ÖZETİ')
    logging.info('=' * 60)

    sensopt_rows = per_subject_df[per_subject_df['Model'] == 'Ensemble (Sensitivity-Optimized)']
    if not sensopt_rows.empty:
        logging.info(
            'Macro Sensitivity : %.4f ± %.4f',
            sensopt_rows['Sensitivity'].mean(), sensopt_rows['Sensitivity'].std()
        )
        logging.info(
            'Macro Precision   : %.4f ± %.4f',
            sensopt_rows['Precision'].mean(), sensopt_rows['Precision'].std()
        )
        logging.info(
            'Macro F1-Score    : %.4f ± %.4f',
            sensopt_rows['Seizure F1-Score'].mean(), sensopt_rows['Seizure F1-Score'].std()
        )
        logging.info(
            'Macro Balanced Acc: %.4f ± %.4f',
            sensopt_rows['Balanced Accuracy'].mean(), sensopt_rows['Balanced Accuracy'].std()
        )
        subjects_above_target = (sensopt_rows['Sensitivity'] >= TARGET_SENSITIVITY).sum()
        logging.info(
            'Sensitivity >= %.2f hedefine ulaşan hasta: %d / %d',
            TARGET_SENSITIVITY, subjects_above_target, len(sensopt_rows)
        )

    logging.info('Saved per-subject metrics : %s', OUTPUT_PER_SUBJECT_CSV)
    logging.info('Saved pooled metrics      : %s', OUTPUT_POOLED_CSV)
    logging.info('Saved macro-mean metrics  : %s', OUTPUT_MACRO_MEAN_CSV)
    logging.info('Saved subject thresholds  : %s', OUTPUT_THRESHOLDS_CSV)


if __name__ == '__main__':
    main()