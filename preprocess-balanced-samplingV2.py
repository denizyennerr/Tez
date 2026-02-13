import os
import gc
import numpy as np
import pandas as pd
import mne
import random
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

# Dataframe file içerisinden filterleyip seizure'Lıları seçiyoruz
seizure_file_list = ( df['file'].tolist())
seizure_set = {f.strip() for f in seizure_file_list}

# 2. edf_paths listesini filtreliyoruz
filtered_paths = [
    path for path in edf_paths
    if os.path.basename(path) in seizure_set
]

# Sonucu görmek için:
print(f"Toplam dosya yolu: {len(edf_paths)}")
print(f"Filtrelenmiş dosya yolu: {len(filtered_paths)}")


def process_and_save_npz(
        edf_list,
        seizure_df,
        output_dir,
        split="train",
        epoch_length=2.0,
        overlap=0.5,
        preictal_exclude=30,
        postictal_exclude=30,
        final_channels=deniz.TARGET_CHANNELS
):
    """
    Production grade preprocessing + balancing pipeline
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    step = epoch_length * (1 - overlap)

    for edf_path in edf_list:

        file_name = os.path.basename(edf_path)
        subject = file_name.split("_")[0]

        print(f"\nProcessing: {file_name}")

        try:
            # ---------------------------
            # 1. Channel Fix
            # ---------------------------
            raw = deniz.fix_eeg_channels_version_2(edf_path, final_channels)
            if raw is None:
                continue

            # ---------------------------
            # 2. Downsample
            # ---------------------------
            raw = deniz.downsample_version2(raw)

            # ---------------------------
            # 3. Filtering
            # ---------------------------
            raw = deniz.apply_filtering_version2(raw)

            # ---------------------------
            # 4. Annotation
            # ---------------------------
            annotations, _, _ = deniz.build_seizure_annotations_for_file(
                seizure_df,
                file_name
            )

            if annotations is not None:
                raw.set_annotations(annotations)

            # ---------------------------
            # 5. Epoch
            # ---------------------------
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=epoch_length,
                overlap=epoch_length - step,
                preload=True,
                verbose=False
            )

            # ---------------------------
            # 6. Label
            # ---------------------------
            labels = deniz.generate_epoch_labels_version2(epochs, raw)
            labels = np.array(labels)

            X = epochs.get_data()

            # ---------------------------
            # 7. Metadata (epoch start times)
            # ---------------------------
            sfreq = raw.info["sfreq"]
            epoch_starts = epochs.events[:, 0] / sfreq

            # ---------------------------
            # 8. BALANCING (TRAIN ONLY)
            # ---------------------------
            if split == "train":

                seizure_idx = np.where(labels == 1)[0]
                non_seizure_idx = np.where(labels == 0)[0]

                # ---- pre/post ictal removal ----
                safe_non_seizure = []

                if raw.annotations is not None:

                    seizure_intervals = [
                        (onset, onset + dur)
                        for onset, dur, desc
                        in zip(raw.annotations.onset,
                               raw.annotations.duration,
                               raw.annotations.description)
                        if desc == "seizure"
                    ]

                    for idx in non_seizure_idx:
                        start = epoch_starts[idx]
                        end = start + epoch_length

                        safe = True

                        for sz_start, sz_end in seizure_intervals:

                            if not (
                                    end <= sz_start - preictal_exclude or
                                    start >= sz_end + postictal_exclude
                            ):
                                safe = False
                                break

                        if safe:
                            safe_non_seizure.append(idx)

                safe_non_seizure = np.array(safe_non_seizure)

                # ---- sample equal amount ----
                if len(seizure_idx) > 0 and len(safe_non_seizure) > 0:

                    sampled_non_seizure = np.random.choice(
                        safe_non_seizure,
                        size=min(len(seizure_idx), len(safe_non_seizure)),
                        replace=False
                    )

                    final_idx = np.concatenate([seizure_idx, sampled_non_seizure])

                else:
                    final_idx = np.arange(len(labels))

            else:
                final_idx = np.arange(len(labels))

            # ---------------------------
            # 9. Final selection
            # ---------------------------
            X_final = X[final_idx]
            y_final = labels[final_idx]
            meta_final = epoch_starts[final_idx]

            # ---------------------------
            # 10. SAVE
            # ---------------------------
            subject_dir = os.path.join(output_dir, subject)
            os.makedirs(subject_dir, exist_ok=True)

            save_name = file_name.replace(".edf", f"_{split}.npz")

            np.savez_compressed(
                os.path.join(subject_dir, save_name),
                x=X_final,
                y=y_final,
                meta=meta_final
            )

            print(
                f"Saved {save_name} | Shape: {X_final.shape} | "
                f"Seizure ratio: {y_final.mean():.3f}"
            )

            # ---------------------------
            # Memory cleanup
            # ---------------------------
            del raw, epochs, X, labels
            gc.collect()

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


process_and_save_npz(
    edf_list=filtered_paths,
    seizure_df=df,
    output_dir="dataset/train",
    split="train"
)

process_and_save_npz(
    edf_list=filtered_paths,
    seizure_df=df,
    output_dir="dataset/val",
    split="val"
)