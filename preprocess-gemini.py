import os
import gc
import numpy as np
import mne
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
seizure_file_list = df['file'].tolist()
seizure_set = {f.strip() for f in seizure_file_list}

# Filter paths
filtered_paths = [
    path for path in edf_paths
    if os.path.basename(path) in seizure_set
]

# ✅ CRITICAL FIX: Patient-level split (prevent data leakage)
patients = list(set([os.path.basename(p).split('_')[0] for p in filtered_paths]))
train_patients, val_patients = train_test_split(
    patients,
    test_size=0.2,
    random_state=42
)

train_paths = [p for p in filtered_paths if os.path.basename(p).split('_')[0] in train_patients]
val_paths = [p for p in filtered_paths if os.path.basename(p).split('_')[0] in val_patients]

print(f"Train patients: {sorted(train_patients)}")
print(f"Val patients: {sorted(val_patients)}")
print(f"Train files: {len(train_paths)}")
print(f"Val files: {len(val_paths)}")


# ...existing imports and setup...

def process_and_save_corrected(
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
    target_dir = os.path.join(output_dir, split)
    os.makedirs(target_dir, exist_ok=True)

    for edf_path in edf_list:
        file_name = os.path.basename(edf_path)
        subject = file_name.split("_")[0]
        print(f"\nProcessing: {file_name} [{split}]")

        try:
            # Load and preprocess
            raw = deniz.fix_eeg_channels_version_2(edf_path, final_channels)
            if raw is None:
                continue

            annotations, _, _ = deniz.build_seizure_annotations_for_file_v2(seizure_df, file_name)
            if annotations:
                raw.set_annotations(annotations)

            raw = deniz.downsample_version2(raw)
            raw = deniz.apply_filtering_version2(raw)

            label_mask = deniz.create_label_mask(
                raw,
                raw.annotations,
                pre_exclude=preictal_exclude,
                post_exclude=postictal_exclude
            )

            # Epoch data
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=epoch_length,
                overlap=overlap,
                preload=True,
                verbose=False
            )
            X = epochs.get_data(copy=True)

            # Stricter epoch labeling
            y_epoch_list = []
            n_samples_per_epoch = X.shape[2]
            seizure_threshold = 0.5

            for event in epochs.events:
                start_samp = event[0]
                end_samp = start_samp + n_samples_per_epoch
                mask_chunk = label_mask[start_samp:end_samp]

                n_seizure = np.sum(mask_chunk == 1)
                n_exclude = np.sum(mask_chunk == -1)

                if n_seizure >= (n_samples_per_epoch * seizure_threshold):
                    label = 1
                elif n_exclude > (n_samples_per_epoch * 0.3):
                    label = -1
                else:
                    label = 0

                y_epoch_list.append(label)

            y = np.array(y_epoch_list)

            # Remove excluded epochs
            valid_mask = (y != -1)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            if len(y_clean) == 0:
                print(f"  ⚠️ No valid data: {file_name}")
                continue

            seizure_idx = np.where(y_clean == 1)[0]
            safe_idx = np.where(y_clean == 0)[0]

            # ✅ CRITICAL FIX: Match training to validation distribution
            if split == "train":
                if len(seizure_idx) > 0:
                    # Target: 1.5-2% seizure prevalence (same as validation)
                    n_seizure = len(seizure_idx)
                    target_prevalence = 0.015  # ✅ Match validation's 1.5%

                    # Calculate required non-seizures
                    n_safe_needed = int(n_seizure * (1 - target_prevalence) / target_prevalence)
                    n_safe_needed = min(n_safe_needed, len(safe_idx))

                    if n_safe_needed > 0:
                        safe_chosen = np.random.choice(safe_idx, size=n_safe_needed, replace=False)
                        final_idx = np.concatenate([seizure_idx, safe_chosen])
                        X_final, y_final = shuffle(X_clean[final_idx], y_clean[final_idx], random_state=42)
                    else:
                        X_final, y_final = X_clean[seizure_idx], y_clean[seizure_idx]
                else:
                    # Non-seizure file: take representative sample
                    n_take = min(300, len(safe_idx))
                    if n_take > 0:
                        chosen_idx = np.random.choice(safe_idx, size=n_take, replace=False)
                        X_final, y_final = X_clean[chosen_idx], y_clean[chosen_idx]
                    else:
                        print(f"  ⚠️ No data: {file_name}")
                        continue
            else:
                # Validation: Keep ALL data (natural distribution)
                X_final, y_final = X_clean, y_clean

            # Save
            subject_dir = os.path.join(target_dir, subject)
            os.makedirs(subject_dir, exist_ok=True)
            save_path = os.path.join(subject_dir, file_name.replace(".edf", f"_{split}.npz"))

            np.savez_compressed(save_path, X=X_final, y=y_final)

            seizure_count = np.sum(y_final == 1)
            print(
                f"  → Saved: {X_final.shape}, Seizures: {seizure_count}/{len(y_final)} ({100 * seizure_count / len(y_final):.2f}%)")

            del raw, epochs, X, label_mask
            gc.collect()

        except Exception as e:
            print(f"❌ Error: {file_name}: {e}")
            import traceback
            traceback.print_exc()


# Run preprocessing
process_and_save_corrected(
    edf_list=train_paths,
    seizure_df=df,
    output_dir='dataset_final_gemini_v3',
    final_channels=FINAL_CHANNELS,
    split='train',
    epoch_length=2.0,
    overlap=0,
    preictal_exclude=30,
    postictal_exclude=30,
)

process_and_save_corrected(
    edf_list=val_paths,
    seizure_df=df,
    output_dir='dataset_final_gemini_v3',
    final_channels=FINAL_CHANNELS,
    split='val',
    epoch_length=2.0,
    overlap=0,
    preictal_exclude=30,
    postictal_exclude=30,
)