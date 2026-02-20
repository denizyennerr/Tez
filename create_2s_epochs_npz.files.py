import numpy as np
import handy.my_utils as deniz
from EEGPreprocessor import EEGPreprocessor
import os
import gc
import pandas as pd

CHB_MIT_PATH = 'data-understanding/data/chb-mit'
preprocessor = EEGPreprocessor(sampling_rate=256, target_sfreq=128)
new_fs = preprocessor.target_sfreq
all_annotations = {}
patient_dirs = sorted([d for d in os.listdir(CHB_MIT_PATH)
                       if os.path.isdir(os.path.join(CHB_MIT_PATH, d))
                       and d.startswith('chb')])

total_seizures = 0
total_files_with_seizures = 0

# Configuration
WINDOW_SIZE = 2  # seconds
OVERLAP = 0.5  # 50% overlap
WINDOW_SAMPLES = WINDOW_SIZE * new_fs  # 2 * 128 = 256 samples
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))

NPZ_OUTPUT_DIR = "processed_npz_files_2s"
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

data_frame_name = 'final_dataset_all_patients_2s.csv'

PATIENTS_TO_USE = ['chb01', 'chb02', 'chb03',
                   'chb04', 'chb05', 'chb06',
                   'chb07', 'chb08', 'chb09',
                   'chb10', 'chb11', 'chb12',
                   'chb13', 'chb14', 'chb15',
                   'chb16', 'chb17', 'chb18',
                   'chb19', 'chb20', 'chb21',
                   'chb22', 'chb23', 'chb24']

FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

print(f"\n👥 Processing {len(patient_dirs)} patients...\n")




for patient in patient_dirs:
    patient_path = os.path.join(CHB_MIT_PATH, patient)
    summary_file = os.path.join(patient_path, f'{patient}-summary.txt')

    if os.path.exists(summary_file):
        seizure_info = deniz.parse_summary_file(summary_file)

        if seizure_info:
            all_annotations[patient] = seizure_info
            n_files = len(seizure_info)
            n_seizures = sum(len(times) for times in seizure_info.values())
            total_files_with_seizures += n_files
            total_seizures += n_seizures

            print(f"   ✓ {patient}: {n_files} files, {n_seizures} seizures")
        else:
            print(f"   - {patient}: No seizures found")
    else:
        print(f"   ✗ {patient}: No summary file")

exclude_list = [
    'chb12/chb12_29.edf',
    'chb12/chb12_27.edf',
    'chb12/chb12_28.edf'
]

# Sözlüğü güvenli bir şekilde güncellemek için iç içe döngü kullanıyoruz
for item in exclude_list:
    # item örneği: 'chb12/chb12_29.edf'
    # folder: 'chb12', filename: 'chb12_29.edf'
    folder, filename = item.split('/')

    # Eğer bu klasör ana sözlükte varsa ve dosya o klasörün içindeyse sil
    if folder in all_annotations and filename in all_annotations[folder]:
        del all_annotations[folder][filename]

# Sonucu doğrulamak için chb12 klasörüne bakalım
print(all_annotations['chb12'].keys())

# Fixed CHATGPT'den npzli olan
# Storage
all_seizure_windows = []
all_normal_windows = []

# NEW → Per-file NPZ output folder


print(f"\n👥 Processing {len(PATIENTS_TO_USE)} patients...")
print("-" * 50)

normal_counter = 0
reporting_patients = {}

for patient in PATIENTS_TO_USE:

    if patient not in all_annotations:
        continue

    patient_seizure = 0
    patient_normal = 0

    for filename, seizure_times in all_annotations[patient].items():

        file_path = os.path.join(CHB_MIT_PATH, patient, filename)

        if not os.path.exists(file_path):
            continue

        try:
            # Load file
            signals_temp, ch_names, fs_temp = deniz.fix_eeg_channels_load_edf(
                file_path=file_path,
                final_channels=FINAL_CHANNELS,
                load_func=deniz.load_edf_file
            )

            if signals_temp is None:
                continue

            # Preprocess
            signals_temp = preprocessor.preprocess(signals_temp)

            n_samples = signals_temp.shape[1]

            # NEW → Per-file storage
            file_seizure_windows = []
            file_normal_windows = []

            # Extract windows
            for start in range(0, n_samples - WINDOW_SAMPLES, STEP_SAMPLES):

                end = start + WINDOW_SAMPLES
                window = signals_temp[:, start:end]

                if window.shape[1] != WINDOW_SAMPLES:
                    continue

                # Label check
                window_start_sec = start / new_fs
                window_end_sec = end / new_fs

                is_seizure = False

                for sz_start, sz_end in seizure_times:

                    if window_start_sec < sz_end and window_end_sec > sz_start:

                        overlap_start = max(window_start_sec, sz_start)
                        overlap_end = min(window_end_sec, sz_end)

                        if (overlap_end - overlap_start) / WINDOW_SIZE >= 0.5:
                            is_seizure = True
                            break

                if is_seizure:
                    all_seizure_windows.append(window.astype(np.float32))
                    file_seizure_windows.append(window.astype(np.float32))
                    patient_seizure += 1

                else:
                    normal_counter += 1
                    if normal_counter % 20 == 0:
                        all_normal_windows.append(window.astype(np.float32))
                        file_normal_windows.append(window.astype(np.float32))
                        patient_normal += 1

            # ================================
            # NEW → Save file-level NPZ
            # ================================
            if len(file_seizure_windows) + len(file_normal_windows) > 0:
                X_file = np.concatenate([
                    np.array(file_seizure_windows, dtype=np.float32),
                    np.array(file_normal_windows, dtype=np.float32)
                ], axis=0)

                y_file = np.concatenate([
                    np.ones(len(file_seizure_windows)),
                    np.zeros(len(file_normal_windows))
                ], axis=0)

                # Shuffle file dataset
                shuffle_idx_file = np.random.permutation(len(X_file))
                X_file = X_file[shuffle_idx_file]
                y_file = y_file[shuffle_idx_file]

                base_name = os.path.splitext(filename)[0]
                npz_filename = f"{patient}_{base_name}.npz"
                npz_path = os.path.join(NPZ_OUTPUT_DIR, npz_filename)

                np.savez_compressed(
                    npz_path,
                    X=X_file,
                    y=y_file,
                    patient=patient,
                    source_file=filename
                )

                del X_file, y_file

            # Clear memory
            del signals_temp
            gc.collect()

        except Exception as e:
            continue

    print(f"   ✓ {patient}: {patient_seizure} seizure, {patient_normal} normal")

    reporting_patients[patient] = {
        'patient_id': patient,
        'seizure_windows': patient_seizure,
        'normal_windows': patient_normal,
        'total_windows': patient_seizure + patient_normal,
        'seizure_percentage': round(
            (patient_seizure / (patient_seizure + patient_normal)) * 100, 2
        ) if (patient_seizure + patient_normal) > 0 else 0
    }

    gc.collect()

print("\n" + "-" * 50)

n_seizure = len(all_seizure_windows)
n_normal_available = len(all_normal_windows)

print(f"📊 Collected: {n_seizure} seizure, {n_normal_available} normal")

# Balance
n_normal_to_use = min(n_normal_available, n_seizure * 2)

np.random.seed(42)

if n_normal_available > n_normal_to_use:

    normal_indices = np.random.choice(
        n_normal_available,
        n_normal_to_use,
        replace=False
    )

    X_normal = np.array(
        [all_normal_windows[i] for i in normal_indices],
        dtype=np.float32
    )

else:
    X_normal = np.array(all_normal_windows, dtype=np.float32)

X_seizure = np.array(all_seizure_windows, dtype=np.float32)

# Clear lists
del all_seizure_windows, all_normal_windows
gc.collect()

# Combine
X = np.concatenate([X_seizure, X_normal], axis=0)
y = np.concatenate([np.ones(len(X_seizure)), np.zeros(len(X_normal))], axis=0)

del X_seizure, X_normal
gc.collect()

# Shuffle
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"\n📈 Final Dataset:")
print(f"   • Total: {len(X)} samples")
print(f"   • Seizure: {int(np.sum(y))} ({100 * np.sum(y) / len(y):.1f}%)")
print(f"   • Normal: {int(len(y) - np.sum(y))} ({100 * (len(y) - np.sum(y)) / len(y):.1f}%)")
print(f"   • Shape: {X.shape}")
print(f"   • Memory: ~{X.nbytes / 1024 / 1024:.1f} MB")

print("\n" + "=" * 60)
print("✅ Dataset ready!")
print("=" * 60)

# 1. Sözlüğü DataFrame'e çevir
df_report = pd.DataFrame.from_dict(reporting_patients, orient='index').reset_index(drop=True)

# 2. Toplam Satırı Ekle (Genel özeti görmek için)
total_row = pd.DataFrame([{
    'patient_id': 'TOTAL (Dataset)',
    'seizure_windows': df_report['seizure_windows'].sum(),
    'normal_windows': df_report['normal_windows'].sum(),
    'total_windows': df_report['total_windows'].sum(),
    'seizure_percentage': round((df_report['seizure_windows'].sum() / df_report['total_windows'].sum()) * 100, 2)
}])

df_final_report = pd.concat([df_report, total_row], ignore_index=True)

# 3. Sonucu Görüntüle
print("\n📊 Patient Data Summary Report:")
print(df_final_report.to_string(index=False))
df_final_report.to_csv(data_frame_name, index=False)

######################

#
# import numpy as np
#
# data = np.load('processed_npz_files_base/chb15_chb15_15.npz')
# X = data['X']
# y = data['y']
# X.shape
# y.shape
#
# import os
#
# path = 'processed_npz_files_base'
# npz_files1 = [f for f in os.listdir(path) if f.endswith('.npz')]
# print(npz_files1)