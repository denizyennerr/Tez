import os
import gc
import numpy as np
import mne
from sklearn.utils import shuffle  # karıştırma için
import pandas as pd
import utility.my_utils as deniz

dataset_path = 'data-understanding/data/chb-mit'
folder_names = deniz.get_folder_names(dataset_path)
edf_paths = deniz.get_edf_paths(dataset_path, folder_names)
FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]
df = pd.read_csv('data-understanding/all_preprocess_pipeline_seizure.csv')
seizure_file_list = (df['file'].tolist())
seizure_set = {f.strip() for f in seizure_file_list}

# 2. edf_paths listesini filtreliyoruz
filtered_paths = [
    path for path in edf_paths
    if os.path.basename(path) in seizure_set
]


def process_and_save_npz(
        edf_list,
        seizure_df,
        output_dir,
        final_channels,
        split='train',
        epoch_length=2.0,
        overlap=0,
        preictal_exclude=30,
        postictal_exclude=30,
):
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True) if split == "train" else os.makedirs(val_dir, exist_ok=True)

    for edf_path in edf_list:
        file_name = os.path.basename(edf_path)
        subject = file_name.split("_")[0]
        print(f"\nProcessing: {file_name} [{split}]")

        try:
            # --- Standart Preprocess ---
            raw = deniz.fix_eeg_channels_version_2(edf_path, final_channels)
            if raw is None: continue
            raw = deniz.downsample_version2(raw)
            raw = deniz.apply_filtering_version2(raw)

            annotations, _, _ = deniz.build_seizure_annotations_for_file_v2(seizure_df, file_name)
            if annotations: raw.set_annotations(annotations)

            epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, overlap=overlap, preload=True,
                                                  verbose=False)

            # --- Yeni Etiketleme Mantığı ---
            # Artık labels: 1 (Seizure), 0 (Safe), -1 (Exclude)
            labels = deniz.generate_epoch_labels_v4(raw, epochs, epoch_length, pre_exclude=preictal_exclude,
                                                    post_exclude=postictal_exclude)
            X = epochs.get_data()

            # Hem Train hem Val için: Yasaklı bölgeleri (-1) tamamen atıyoruz
            valid_mask = (labels != -1)
            X_clean = X[valid_mask]
            y_clean = labels[valid_mask]

            if len(y_clean) == 0:
                print(f"  ⚠️ Dosya tamamen exclude edildi: {file_name}")
                continue

            # --- Train / Val Karar Mekanizması ---
            if split == "train":
                seizure_idx = np.where(y_clean == 1)[0]
                safe_non_seizure_idx = np.where(y_clean == 0)[0]
                n_seizure = len(seizure_idx)

                if n_seizure > 0:
                    # Nöbet varsa: 1:1 oranında güvenli non-seizure örnekle
                    n_take = min(n_seizure, len(safe_non_seizure_idx))
                    step = max(1, len(safe_non_seizure_idx) // n_take)
                    sampled_non_idx = safe_non_seizure_idx[::step][:n_take]

                    final_idx = np.concatenate([seizure_idx, sampled_non_idx])
                    X_final, y_final = X_clean[final_idx], y_clean[final_idx]
                    X_final, y_final = shuffle(X_final, y_final, random_state=42)
                else:
                    # Nöbet yoksa: Train setine binlerce non-seizure eklememek için
                    # sabit (örneğin 50-100 adet) temsilci örnek al veya dosyayı geç.
                    # Şimdilik örnek alıyoruz:
                    n_dummy = min(50, len(safe_non_seizure_idx))
                    step = max(1, len(safe_non_seizure_idx) // n_dummy)
                    final_idx = safe_non_seizure_idx[::step][:n_dummy]
                    X_final, y_final = X_clean[final_idx], y_clean[final_idx]

            else:
                # Val setinde dengeleme yapma ama yasaklı bölgeleri (maske ile) temizlemiş olduk
                X_final, y_final = X_clean, y_clean

            # --- Kayıt ---
            target_dir = train_dir if split == "train" else val_dir
            subject_dir = os.path.join(target_dir, subject)
            os.makedirs(subject_dir, exist_ok=True)

            save_path = os.path.join(subject_dir, file_name.replace(".edf", f"_{split}.npz"))
            np.savez_compressed(save_path, X=X_final, y=y_final, subject=subject, file_name=file_name)

            print(f"  → Final shape: {X_final.shape} | Seizure ratio: {y_final.mean():.3f}")
            del raw, epochs, X, X_clean, X_final;
            gc.collect()

        except Exception as e:
            print(f"❌ Error: {str(e)}")


process_and_save_npz(
    edf_list=filtered_paths,
    seizure_df=df,
    output_dir='dataset_final_gemini',
    final_channels=FINAL_CHANNELS,
    split='train',
    epoch_length=2.0,
    overlap=0,
    preictal_exclude=30,  # saniye
    postictal_exclude=30,  # saniye
)

process_and_save_npz(
    edf_list=filtered_paths,
    seizure_df=df,
    output_dir='dataset_final_gemini',
    final_channels=FINAL_CHANNELS,
    split='val',
    epoch_length=2.0,
    overlap=0,
    preictal_exclude=30,  # saniye
    postictal_exclude=30,  # saniye
)

# import numpy as np
# path = "dataset_final/train/chb02/chb02_16_train.npz"
# npz = np.load(path)
# npz['X']
# np.count_nonzero(npz['y'])
