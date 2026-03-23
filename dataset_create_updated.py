import numpy as np
import handy.my_utils as deniz
from EEGPreprocessor import EEGPreprocessor
import os
import gc
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
CHB_MIT_PATH = 'data-understanding/data/chb-mit'
NPZ_OUTPUT_DIR = "processed_master_datasets"
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

# ---> CHANGE THIS VALUE to generate datasets for 0.5, 1.0, 2.0, 4.0, 5.0, 10.0
WINDOW_SIZE = 0.5

# RULE 1: Zero Overlap. Windows must be strictly contiguous.
OVERLAP = 0.0

# RULE 2: Zero Padding reference. All EDF files must pad up to a multiple of 10s.
LARGEST_WINDOW_REF = 10.0

preprocessor = EEGPreprocessor(sampling_rate=256, target_sfreq=128)
new_fs = preprocessor.target_sfreq

WINDOW_SAMPLES = int(round(WINDOW_SIZE * new_fs))
STEP_SAMPLES = int(round(WINDOW_SAMPLES * (1 - OVERLAP)))

PATIENTS_TO_USE = [
    'chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
    'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
    'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
    'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24'
]

FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# =============================================================================
# SUMMARY PARSING
# =============================================================================
all_annotations = {}
patient_dirs = sorted(
    [d for d in os.listdir(CHB_MIT_PATH) if os.path.isdir(os.path.join(CHB_MIT_PATH, d)) and d.startswith('chb')])

print(f"\n👥 Parsing summary files for {len(patient_dirs)} patients...\n")
for patient in patient_dirs:
    summary_file = os.path.join(CHB_MIT_PATH, patient, f'{patient}-summary.txt')
    if os.path.exists(summary_file):
        seizure_info = deniz.parse_summary_file(summary_file)
        if seizure_info:
            all_annotations[patient] = seizure_info

exclude_list = ['chb12/chb12_29.edf', 'chb12/chb12_27.edf', 'chb12/chb12_28.edf']
for item in exclude_list:
    folder, filename = item.split('/')
    if folder in all_annotations and filename in all_annotations[folder]:
        del all_annotations[folder][filename]

# =============================================================================
# DATA EXTRACTION
# =============================================================================
# We will collect everything sequentially into these lists
master_X = []
master_y = []
master_s = []
master_file_id = []

print(f"\n🚀 Extracting strictly aligned {WINDOW_SIZE}s dataset...")
print("-" * 50)

for patient in PATIENTS_TO_USE:
    if patient not in all_annotations:
        continue

    patient_seizure_count = 0
    patient_normal_count = 0

    # Ensure files are processed in alphabetical order to maintain chronological integrity
    for filename in sorted(all_annotations[patient].keys()):
        seizure_times = all_annotations[patient][filename]
        file_path = os.path.join(CHB_MIT_PATH, patient, filename)

        if not os.path.exists(file_path):
            continue

        try:
            # 1. Load File
            signals_temp, ch_names, fs_temp = deniz.fix_eeg_channels_load_edf(
                file_path=file_path,
                final_channels=FINAL_CHANNELS,
                load_func=deniz.load_edf_file
            )
            if signals_temp is None:
                continue

            # 2. Preprocess (Resampling, filtering, etc.)
            signals_temp = preprocessor.preprocess(signals_temp)
            n_samples = signals_temp.shape[1]

            # -----------------------------------------------------------------
            # RULE 2: ZERO PADDING
            # -----------------------------------------------------------------
            duration_sec = n_samples / new_fs
            remainder = duration_sec % LARGEST_WINDOW_REF

            if remainder > 0:
                pad_sec = LARGEST_WINDOW_REF - remainder
                pad_samples = int(round(pad_sec * new_fs))

                padding = np.zeros((signals_temp.shape[0], pad_samples), dtype=np.float32)
                signals_temp = np.concatenate([signals_temp, padding], axis=1)

            n_samples_padded = signals_temp.shape[1]

            # 3. Extract Windows Sequentially
            for start in range(0, n_samples_padded - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                end = start + WINDOW_SAMPLES
                window = signals_temp[:, start:end]

                window_start_sec = start / new_fs
                window_end_sec = end / new_fs

                # -------------------------------------------------------------
                # ENSEMBLE ALIGNMENT RULE: NEVER DROP WINDOWS
                # -------------------------------------------------------------
                overlap_duration = 0.0
                for sz_start, sz_end in seizure_times:
                    overlap_start = max(window_start_sec, sz_start)
                    overlap_end = min(window_end_sec, sz_end)

                    if overlap_end > overlap_start:
                        overlap_duration += (overlap_end - overlap_start)

                overlap_ratio = overlap_duration / WINDOW_SIZE

                # Kesin Karar: %50'dan büyük eşitse 1, değilse 0. Hiçbir pencere çöpe atılmaz.
                is_seizure = (overlap_ratio >= 0.50)

                master_X.append(window.astype(np.float32))
                master_y.append(1 if is_seizure else 0)
                master_s.append(patient)
                master_file_id.append(filename)

                if is_seizure:
                    patient_seizure_count += 1
                else:
                    patient_normal_count += 1

            # Clear memory
            del signals_temp
            gc.collect()

        except Exception as e:
            print(f"   [!] Error processing {filename}: {e}")
            continue

    print(f"   ✓ {patient}: {patient_seizure_count} seizure, {patient_normal_count} normal windows (100% Retained)")
    gc.collect()

# =============================================================================
# SAVING THE DATASET
# =============================================================================
print("\n" + "-" * 50)
print("📦 Compiling and saving the master dataset...")

# Initial shape will be (N, Channels, Window_Samples)
X = np.array(master_X, dtype=np.float32)
y = np.array(master_y, dtype=np.int8)
s = np.array(master_s, dtype=str)
file_id = np.array(master_file_id, dtype=str)

# Clear lists to free RAM before transposing and saving
del master_X, master_y, master_s, master_file_id
gc.collect()

# -----------------------------------------------------------------------------
# TRANSPOSING THE DATA
# -----------------------------------------------------------------------------
X = np.transpose(X, (2, 0, 1))

print(f"\n📈 Final {WINDOW_SIZE}s Dataset:")
print(f"   • Total Windows: {len(y)}")
print(f"   • Shape of X:    {X.shape} (Window Size, Memory Size, Channels)")
print(f"   • Memory Size:   ~{X.nbytes / 1024 / 1024 / 1024:.2f} GB")

out_file = os.path.join(NPZ_OUTPUT_DIR, f"master_dataset_{WINDOW_SIZE}s.npz")
np.savez_compressed(
    out_file,
    X=X,
    y=y,
    s=s,
    file_id=file_id
)

del X, y, file_id
print("\n" + "=" * 60)
print(f"✅ Master dataset securely saved to {out_file}!")
print("=" * 60)
#############################################################################################################
import numpy as np
import handy.my_utils as deniz
from EEGPreprocessor import EEGPreprocessor
import os
import gc
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
CHB_MIT_PATH = 'data-understanding/data/chb-mit'
NPZ_OUTPUT_DIR = "processed_master_datasets"
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

# ---> CHANGE THIS VALUE to generate datasets for 0.5, 1.0, 2.0, 4.0, 5.0, 10.0
WINDOW_SIZE = 1.0

# RULE 1: Zero Overlap. Windows must be strictly contiguous.
OVERLAP = 0.0

# RULE 2: Zero Padding reference. All EDF files must pad up to a multiple of 10s.
LARGEST_WINDOW_REF = 10.0

preprocessor = EEGPreprocessor(sampling_rate=256, target_sfreq=128)
new_fs = preprocessor.target_sfreq

WINDOW_SAMPLES = int(round(WINDOW_SIZE * new_fs))
STEP_SAMPLES = int(round(WINDOW_SAMPLES * (1 - OVERLAP)))

PATIENTS_TO_USE = [
    'chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
    'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
    'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
    'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24'
]

FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# =============================================================================
# SUMMARY PARSING
# =============================================================================
all_annotations = {}
patient_dirs = sorted(
    [d for d in os.listdir(CHB_MIT_PATH) if os.path.isdir(os.path.join(CHB_MIT_PATH, d)) and d.startswith('chb')])

print(f"\n👥 Parsing summary files for {len(patient_dirs)} patients...\n")
for patient in patient_dirs:
    summary_file = os.path.join(CHB_MIT_PATH, patient, f'{patient}-summary.txt')
    if os.path.exists(summary_file):
        seizure_info = deniz.parse_summary_file(summary_file)
        if seizure_info:
            all_annotations[patient] = seizure_info

exclude_list = ['chb12/chb12_29.edf', 'chb12/chb12_27.edf', 'chb12/chb12_28.edf']
for item in exclude_list:
    folder, filename = item.split('/')
    if folder in all_annotations and filename in all_annotations[folder]:
        del all_annotations[folder][filename]

# =============================================================================
# DATA EXTRACTION
# =============================================================================
# We will collect everything sequentially into these lists
master_X = []
master_y = []
master_s = []
master_file_id = []

print(f"\n🚀 Extracting strictly aligned {WINDOW_SIZE}s dataset...")
print("-" * 50)

for patient in PATIENTS_TO_USE:
    if patient not in all_annotations:
        continue

    patient_seizure_count = 0
    patient_normal_count = 0

    # Ensure files are processed in alphabetical order to maintain chronological integrity
    for filename in sorted(all_annotations[patient].keys()):
        seizure_times = all_annotations[patient][filename]
        file_path = os.path.join(CHB_MIT_PATH, patient, filename)

        if not os.path.exists(file_path):
            continue

        try:
            # 1. Load File
            signals_temp, ch_names, fs_temp = deniz.fix_eeg_channels_load_edf(
                file_path=file_path,
                final_channels=FINAL_CHANNELS,
                load_func=deniz.load_edf_file
            )
            if signals_temp is None:
                continue

            # 2. Preprocess (Resampling, filtering, etc.)
            signals_temp = preprocessor.preprocess(signals_temp)
            n_samples = signals_temp.shape[1]

            # -----------------------------------------------------------------
            # RULE 2: ZERO PADDING
            # -----------------------------------------------------------------
            duration_sec = n_samples / new_fs
            remainder = duration_sec % LARGEST_WINDOW_REF

            if remainder > 0:
                pad_sec = LARGEST_WINDOW_REF - remainder
                pad_samples = int(round(pad_sec * new_fs))

                padding = np.zeros((signals_temp.shape[0], pad_samples), dtype=np.float32)
                signals_temp = np.concatenate([signals_temp, padding], axis=1)

            n_samples_padded = signals_temp.shape[1]

            # 3. Extract Windows Sequentially
            for start in range(0, n_samples_padded - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                end = start + WINDOW_SAMPLES
                window = signals_temp[:, start:end]

                window_start_sec = start / new_fs
                window_end_sec = end / new_fs

                # -------------------------------------------------------------
                # ENSEMBLE ALIGNMENT RULE: NEVER DROP WINDOWS
                # -------------------------------------------------------------
                overlap_duration = 0.0
                for sz_start, sz_end in seizure_times:
                    overlap_start = max(window_start_sec, sz_start)
                    overlap_end = min(window_end_sec, sz_end)

                    if overlap_end > overlap_start:
                        overlap_duration += (overlap_end - overlap_start)

                overlap_ratio = overlap_duration / WINDOW_SIZE

                # Kesin Karar: %50'dan büyük eşitse 1, değilse 0. Hiçbir pencere çöpe atılmaz.
                is_seizure = (overlap_ratio >= 0.50)

                master_X.append(window.astype(np.float32))
                master_y.append(1 if is_seizure else 0)
                master_s.append(patient)
                master_file_id.append(filename)

                if is_seizure:
                    patient_seizure_count += 1
                else:
                    patient_normal_count += 1

            # Clear memory
            del signals_temp
            gc.collect()

        except Exception as e:
            print(f"   [!] Error processing {filename}: {e}")
            continue

    print(f"   ✓ {patient}: {patient_seizure_count} seizure, {patient_normal_count} normal windows (100% Retained)")
    gc.collect()

# =============================================================================
# SAVING THE DATASET
# =============================================================================
print("\n" + "-" * 50)
print("📦 Compiling and saving the master dataset...")

# Initial shape will be (N, Channels, Window_Samples)
X = np.array(master_X, dtype=np.float32)
y = np.array(master_y, dtype=np.int8)
s = np.array(master_s, dtype=str)
file_id = np.array(master_file_id, dtype=str)

# Clear lists to free RAM before transposing and saving
del master_X, master_y, master_s, master_file_id
gc.collect()

# -----------------------------------------------------------------------------
# TRANSPOSING THE DATA
# -----------------------------------------------------------------------------
X = np.transpose(X, (2, 0, 1))

print(f"\n📈 Final {WINDOW_SIZE}s Dataset:")
print(f"   • Total Windows: {len(y)}")
print(f"   • Shape of X:    {X.shape} (Window Size, Memory Size, Channels)")
print(f"   • Memory Size:   ~{X.nbytes / 1024 / 1024 / 1024:.2f} GB")

out_file = os.path.join(NPZ_OUTPUT_DIR, f"master_dataset_{WINDOW_SIZE}s.npz")
np.savez_compressed(
    out_file,
    X=X,
    y=y,
    s=s,
    file_id=file_id
)

del X, y, file_id
print("\n" + "=" * 60)
print(f"✅ Master dataset securely saved to {out_file}!")
print("=" * 60)
########################################################################################33
import numpy as np
import handy.my_utils as deniz
from EEGPreprocessor import EEGPreprocessor
import os
import gc
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
CHB_MIT_PATH = 'data-understanding/data/chb-mit'
NPZ_OUTPUT_DIR = "processed_master_datasets"
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

# ---> CHANGE THIS VALUE to generate datasets for 0.5, 1.0, 2.0, 4.0, 5.0, 10.0
WINDOW_SIZE = 2.0

# RULE 1: Zero Overlap. Windows must be strictly contiguous.
OVERLAP = 0.0

# RULE 2: Zero Padding reference. All EDF files must pad up to a multiple of 10s.
LARGEST_WINDOW_REF = 10.0

preprocessor = EEGPreprocessor(sampling_rate=256, target_sfreq=128)
new_fs = preprocessor.target_sfreq

WINDOW_SAMPLES = int(round(WINDOW_SIZE * new_fs))
STEP_SAMPLES = int(round(WINDOW_SAMPLES * (1 - OVERLAP)))

PATIENTS_TO_USE = [
    'chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
    'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
    'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
    'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24'
]

FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# =============================================================================
# SUMMARY PARSING
# =============================================================================
all_annotations = {}
patient_dirs = sorted(
    [d for d in os.listdir(CHB_MIT_PATH) if os.path.isdir(os.path.join(CHB_MIT_PATH, d)) and d.startswith('chb')])

print(f"\n👥 Parsing summary files for {len(patient_dirs)} patients...\n")
for patient in patient_dirs:
    summary_file = os.path.join(CHB_MIT_PATH, patient, f'{patient}-summary.txt')
    if os.path.exists(summary_file):
        seizure_info = deniz.parse_summary_file(summary_file)
        if seizure_info:
            all_annotations[patient] = seizure_info

exclude_list = ['chb12/chb12_29.edf', 'chb12/chb12_27.edf', 'chb12/chb12_28.edf']
for item in exclude_list:
    folder, filename = item.split('/')
    if folder in all_annotations and filename in all_annotations[folder]:
        del all_annotations[folder][filename]

# =============================================================================
# DATA EXTRACTION
# =============================================================================
# We will collect everything sequentially into these lists
master_X = []
master_y = []
master_s = []
master_file_id = []

print(f"\n🚀 Extracting strictly aligned {WINDOW_SIZE}s dataset...")
print("-" * 50)

for patient in PATIENTS_TO_USE:
    if patient not in all_annotations:
        continue

    patient_seizure_count = 0
    patient_normal_count = 0

    # Ensure files are processed in alphabetical order to maintain chronological integrity
    for filename in sorted(all_annotations[patient].keys()):
        seizure_times = all_annotations[patient][filename]
        file_path = os.path.join(CHB_MIT_PATH, patient, filename)

        if not os.path.exists(file_path):
            continue

        try:
            # 1. Load File
            signals_temp, ch_names, fs_temp = deniz.fix_eeg_channels_load_edf(
                file_path=file_path,
                final_channels=FINAL_CHANNELS,
                load_func=deniz.load_edf_file
            )
            if signals_temp is None:
                continue

            # 2. Preprocess (Resampling, filtering, etc.)
            signals_temp = preprocessor.preprocess(signals_temp)
            n_samples = signals_temp.shape[1]

            # -----------------------------------------------------------------
            # RULE 2: ZERO PADDING
            # -----------------------------------------------------------------
            duration_sec = n_samples / new_fs
            remainder = duration_sec % LARGEST_WINDOW_REF

            if remainder > 0:
                pad_sec = LARGEST_WINDOW_REF - remainder
                pad_samples = int(round(pad_sec * new_fs))

                padding = np.zeros((signals_temp.shape[0], pad_samples), dtype=np.float32)
                signals_temp = np.concatenate([signals_temp, padding], axis=1)

            n_samples_padded = signals_temp.shape[1]

            # 3. Extract Windows Sequentially
            for start in range(0, n_samples_padded - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                end = start + WINDOW_SAMPLES
                window = signals_temp[:, start:end]

                window_start_sec = start / new_fs
                window_end_sec = end / new_fs

                # -------------------------------------------------------------
                # ENSEMBLE ALIGNMENT RULE: NEVER DROP WINDOWS
                # -------------------------------------------------------------
                overlap_duration = 0.0
                for sz_start, sz_end in seizure_times:
                    overlap_start = max(window_start_sec, sz_start)
                    overlap_end = min(window_end_sec, sz_end)

                    if overlap_end > overlap_start:
                        overlap_duration += (overlap_end - overlap_start)

                overlap_ratio = overlap_duration / WINDOW_SIZE

                # Kesin Karar: %50'dan büyük eşitse 1, değilse 0. Hiçbir pencere çöpe atılmaz.
                is_seizure = (overlap_ratio >= 0.50)

                master_X.append(window.astype(np.float32))
                master_y.append(1 if is_seizure else 0)
                master_s.append(patient)
                master_file_id.append(filename)

                if is_seizure:
                    patient_seizure_count += 1
                else:
                    patient_normal_count += 1

            # Clear memory
            del signals_temp
            gc.collect()

        except Exception as e:
            print(f"   [!] Error processing {filename}: {e}")
            continue

    print(f"   ✓ {patient}: {patient_seizure_count} seizure, {patient_normal_count} normal windows (100% Retained)")
    gc.collect()

# =============================================================================
# SAVING THE DATASET
# =============================================================================
print("\n" + "-" * 50)
print("📦 Compiling and saving the master dataset...")

# Initial shape will be (N, Channels, Window_Samples)
X = np.array(master_X, dtype=np.float32)
y = np.array(master_y, dtype=np.int8)
s = np.array(master_s, dtype=str)
file_id = np.array(master_file_id, dtype=str)

# Clear lists to free RAM before transposing and saving
del master_X, master_y, master_s, master_file_id
gc.collect()

# -----------------------------------------------------------------------------
# TRANSPOSING THE DATA
# -----------------------------------------------------------------------------
X = np.transpose(X, (2, 0, 1))

print(f"\n📈 Final {WINDOW_SIZE}s Dataset:")
print(f"   • Total Windows: {len(y)}")
print(f"   • Shape of X:    {X.shape} (Window Size, Memory Size, Channels)")
print(f"   • Memory Size:   ~{X.nbytes / 1024 / 1024 / 1024:.2f} GB")

out_file = os.path.join(NPZ_OUTPUT_DIR, f"master_dataset_{WINDOW_SIZE}s.npz")
np.savez_compressed(
    out_file,
    X=X,
    y=y,
    s=s,
    file_id=file_id
)

del X, y, file_id
print("\n" + "=" * 60)
print(f"✅ Master dataset securely saved to {out_file}!")
print("=" * 60)
########################################################################################################
import numpy as np
import handy.my_utils as deniz
from EEGPreprocessor import EEGPreprocessor
import os
import gc
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
CHB_MIT_PATH = 'data-understanding/data/chb-mit'
NPZ_OUTPUT_DIR = "processed_master_datasets"
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

# ---> CHANGE THIS VALUE to generate datasets for 0.5, 1.0, 2.0, 4.0, 5.0, 10.0
WINDOW_SIZE = 4.0

# RULE 1: Zero Overlap. Windows must be strictly contiguous.
OVERLAP = 0.0

# RULE 2: Zero Padding reference. All EDF files must pad up to a multiple of 10s.
LARGEST_WINDOW_REF = 10.0

preprocessor = EEGPreprocessor(sampling_rate=256, target_sfreq=128)
new_fs = preprocessor.target_sfreq

WINDOW_SAMPLES = int(round(WINDOW_SIZE * new_fs))
STEP_SAMPLES = int(round(WINDOW_SAMPLES * (1 - OVERLAP)))

PATIENTS_TO_USE = [
    'chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
    'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
    'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
    'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24'
]

FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# =============================================================================
# SUMMARY PARSING
# =============================================================================
all_annotations = {}
patient_dirs = sorted(
    [d for d in os.listdir(CHB_MIT_PATH) if os.path.isdir(os.path.join(CHB_MIT_PATH, d)) and d.startswith('chb')])

print(f"\n👥 Parsing summary files for {len(patient_dirs)} patients...\n")
for patient in patient_dirs:
    summary_file = os.path.join(CHB_MIT_PATH, patient, f'{patient}-summary.txt')
    if os.path.exists(summary_file):
        seizure_info = deniz.parse_summary_file(summary_file)
        if seizure_info:
            all_annotations[patient] = seizure_info

exclude_list = ['chb12/chb12_29.edf', 'chb12/chb12_27.edf', 'chb12/chb12_28.edf']
for item in exclude_list:
    folder, filename = item.split('/')
    if folder in all_annotations and filename in all_annotations[folder]:
        del all_annotations[folder][filename]

# =============================================================================
# DATA EXTRACTION
# =============================================================================
# We will collect everything sequentially into these lists
master_X = []
master_y = []
master_s = []
master_file_id = []

print(f"\n🚀 Extracting strictly aligned {WINDOW_SIZE}s dataset...")
print("-" * 50)

for patient in PATIENTS_TO_USE:
    if patient not in all_annotations:
        continue

    patient_seizure_count = 0
    patient_normal_count = 0

    # Ensure files are processed in alphabetical order to maintain chronological integrity
    for filename in sorted(all_annotations[patient].keys()):
        seizure_times = all_annotations[patient][filename]
        file_path = os.path.join(CHB_MIT_PATH, patient, filename)

        if not os.path.exists(file_path):
            continue

        try:
            # 1. Load File
            signals_temp, ch_names, fs_temp = deniz.fix_eeg_channels_load_edf(
                file_path=file_path,
                final_channels=FINAL_CHANNELS,
                load_func=deniz.load_edf_file
            )
            if signals_temp is None:
                continue

            # 2. Preprocess (Resampling, filtering, etc.)
            signals_temp = preprocessor.preprocess(signals_temp)
            n_samples = signals_temp.shape[1]

            # -----------------------------------------------------------------
            # RULE 2: ZERO PADDING
            # -----------------------------------------------------------------
            duration_sec = n_samples / new_fs
            remainder = duration_sec % LARGEST_WINDOW_REF

            if remainder > 0:
                pad_sec = LARGEST_WINDOW_REF - remainder
                pad_samples = int(round(pad_sec * new_fs))

                padding = np.zeros((signals_temp.shape[0], pad_samples), dtype=np.float32)
                signals_temp = np.concatenate([signals_temp, padding], axis=1)

            n_samples_padded = signals_temp.shape[1]

            # 3. Extract Windows Sequentially
            for start in range(0, n_samples_padded - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                end = start + WINDOW_SAMPLES
                window = signals_temp[:, start:end]

                window_start_sec = start / new_fs
                window_end_sec = end / new_fs

                # -------------------------------------------------------------
                # ENSEMBLE ALIGNMENT RULE: NEVER DROP WINDOWS
                # -------------------------------------------------------------
                overlap_duration = 0.0
                for sz_start, sz_end in seizure_times:
                    overlap_start = max(window_start_sec, sz_start)
                    overlap_end = min(window_end_sec, sz_end)

                    if overlap_end > overlap_start:
                        overlap_duration += (overlap_end - overlap_start)

                overlap_ratio = overlap_duration / WINDOW_SIZE

                # Kesin Karar: %50'dan büyük eşitse 1, değilse 0. Hiçbir pencere çöpe atılmaz.
                is_seizure = (overlap_ratio >= 0.50)

                master_X.append(window.astype(np.float32))
                master_y.append(1 if is_seizure else 0)
                master_s.append(patient)
                master_file_id.append(filename)

                if is_seizure:
                    patient_seizure_count += 1
                else:
                    patient_normal_count += 1

            # Clear memory
            del signals_temp
            gc.collect()

        except Exception as e:
            print(f"   [!] Error processing {filename}: {e}")
            continue

    print(f"   ✓ {patient}: {patient_seizure_count} seizure, {patient_normal_count} normal windows (100% Retained)")
    gc.collect()

# =============================================================================
# SAVING THE DATASET
# =============================================================================
print("\n" + "-" * 50)
print("📦 Compiling and saving the master dataset...")

# Initial shape will be (N, Channels, Window_Samples)
X = np.array(master_X, dtype=np.float32)
y = np.array(master_y, dtype=np.int8)
s = np.array(master_s, dtype=str)
file_id = np.array(master_file_id, dtype=str)

# Clear lists to free RAM before transposing and saving
del master_X, master_y, master_s, master_file_id
gc.collect()

# -----------------------------------------------------------------------------
# TRANSPOSING THE DATA
# -----------------------------------------------------------------------------
X = np.transpose(X, (2, 0, 1))

print(f"\n📈 Final {WINDOW_SIZE}s Dataset:")
print(f"   • Total Windows: {len(y)}")
print(f"   • Shape of X:    {X.shape} (Window Size, Memory Size, Channels)")
print(f"   • Memory Size:   ~{X.nbytes / 1024 / 1024 / 1024:.2f} GB")

out_file = os.path.join(NPZ_OUTPUT_DIR, f"master_dataset_{WINDOW_SIZE}s.npz")
np.savez_compressed(
    out_file,
    X=X,
    y=y,
    s=s,
    file_id=file_id
)

del X, y, file_id
print("\n" + "=" * 60)
print(f"✅ Master dataset securely saved to {out_file}!")
print("=" * 60)
########################################################################################################
import numpy as np
import handy.my_utils as deniz
from EEGPreprocessor import EEGPreprocessor
import os
import gc
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
CHB_MIT_PATH = 'data-understanding/data/chb-mit'
NPZ_OUTPUT_DIR = "processed_master_datasets"
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

# ---> CHANGE THIS VALUE to generate datasets for 0.5, 1.0, 2.0, 4.0, 5.0, 10.0
WINDOW_SIZE = 5.0

# RULE 1: Zero Overlap. Windows must be strictly contiguous.
OVERLAP = 0.0

# RULE 2: Zero Padding reference. All EDF files must pad up to a multiple of 10s.
LARGEST_WINDOW_REF = 10.0

preprocessor = EEGPreprocessor(sampling_rate=256, target_sfreq=128)
new_fs = preprocessor.target_sfreq

WINDOW_SAMPLES = int(round(WINDOW_SIZE * new_fs))
STEP_SAMPLES = int(round(WINDOW_SAMPLES * (1 - OVERLAP)))

PATIENTS_TO_USE = [
    'chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
    'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
    'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
    'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24'
]

FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# =============================================================================
# SUMMARY PARSING
# =============================================================================
all_annotations = {}
patient_dirs = sorted(
    [d for d in os.listdir(CHB_MIT_PATH) if os.path.isdir(os.path.join(CHB_MIT_PATH, d)) and d.startswith('chb')])

print(f"\n👥 Parsing summary files for {len(patient_dirs)} patients...\n")
for patient in patient_dirs:
    summary_file = os.path.join(CHB_MIT_PATH, patient, f'{patient}-summary.txt')
    if os.path.exists(summary_file):
        seizure_info = deniz.parse_summary_file(summary_file)
        if seizure_info:
            all_annotations[patient] = seizure_info

exclude_list = ['chb12/chb12_29.edf', 'chb12/chb12_27.edf', 'chb12/chb12_28.edf']
for item in exclude_list:
    folder, filename = item.split('/')
    if folder in all_annotations and filename in all_annotations[folder]:
        del all_annotations[folder][filename]

# =============================================================================
# DATA EXTRACTION
# =============================================================================
# We will collect everything sequentially into these lists
master_X = []
master_y = []
master_s = []
master_file_id = []

print(f"\n🚀 Extracting strictly aligned {WINDOW_SIZE}s dataset...")
print("-" * 50)

for patient in PATIENTS_TO_USE:
    if patient not in all_annotations:
        continue

    patient_seizure_count = 0
    patient_normal_count = 0

    # Ensure files are processed in alphabetical order to maintain chronological integrity
    for filename in sorted(all_annotations[patient].keys()):
        seizure_times = all_annotations[patient][filename]
        file_path = os.path.join(CHB_MIT_PATH, patient, filename)

        if not os.path.exists(file_path):
            continue

        try:
            # 1. Load File
            signals_temp, ch_names, fs_temp = deniz.fix_eeg_channels_load_edf(
                file_path=file_path,
                final_channels=FINAL_CHANNELS,
                load_func=deniz.load_edf_file
            )
            if signals_temp is None:
                continue

            # 2. Preprocess (Resampling, filtering, etc.)
            signals_temp = preprocessor.preprocess(signals_temp)
            n_samples = signals_temp.shape[1]

            # -----------------------------------------------------------------
            # RULE 2: ZERO PADDING
            # -----------------------------------------------------------------
            duration_sec = n_samples / new_fs
            remainder = duration_sec % LARGEST_WINDOW_REF

            if remainder > 0:
                pad_sec = LARGEST_WINDOW_REF - remainder
                pad_samples = int(round(pad_sec * new_fs))

                padding = np.zeros((signals_temp.shape[0], pad_samples), dtype=np.float32)
                signals_temp = np.concatenate([signals_temp, padding], axis=1)

            n_samples_padded = signals_temp.shape[1]

            # 3. Extract Windows Sequentially
            for start in range(0, n_samples_padded - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                end = start + WINDOW_SAMPLES
                window = signals_temp[:, start:end]

                window_start_sec = start / new_fs
                window_end_sec = end / new_fs

                # -------------------------------------------------------------
                # ENSEMBLE ALIGNMENT RULE: NEVER DROP WINDOWS
                # -------------------------------------------------------------
                overlap_duration = 0.0
                for sz_start, sz_end in seizure_times:
                    overlap_start = max(window_start_sec, sz_start)
                    overlap_end = min(window_end_sec, sz_end)

                    if overlap_end > overlap_start:
                        overlap_duration += (overlap_end - overlap_start)

                overlap_ratio = overlap_duration / WINDOW_SIZE

                # Kesin Karar: %50'dan büyük eşitse 1, değilse 0. Hiçbir pencere çöpe atılmaz.
                is_seizure = (overlap_ratio >= 0.50)

                master_X.append(window.astype(np.float32))
                master_y.append(1 if is_seizure else 0)
                master_s.append(patient)
                master_file_id.append(filename)

                if is_seizure:
                    patient_seizure_count += 1
                else:
                    patient_normal_count += 1

            # Clear memory
            del signals_temp
            gc.collect()

        except Exception as e:
            print(f"   [!] Error processing {filename}: {e}")
            continue

    print(f"   ✓ {patient}: {patient_seizure_count} seizure, {patient_normal_count} normal windows (100% Retained)")
    gc.collect()

# =============================================================================
# SAVING THE DATASET
# =============================================================================
print("\n" + "-" * 50)
print("📦 Compiling and saving the master dataset...")

# Initial shape will be (N, Channels, Window_Samples)
X = np.array(master_X, dtype=np.float32)
y = np.array(master_y, dtype=np.int8)
s = np.array(master_s, dtype=str)
file_id = np.array(master_file_id, dtype=str)

# Clear lists to free RAM before transposing and saving
del master_X, master_y, master_s, master_file_id
gc.collect()

# -----------------------------------------------------------------------------
# TRANSPOSING THE DATA
# -----------------------------------------------------------------------------
X = np.transpose(X, (2, 0, 1))

print(f"\n📈 Final {WINDOW_SIZE}s Dataset:")
print(f"   • Total Windows: {len(y)}")
print(f"   • Shape of X:    {X.shape} (Window Size, Memory Size, Channels)")
print(f"   • Memory Size:   ~{X.nbytes / 1024 / 1024 / 1024:.2f} GB")

out_file = os.path.join(NPZ_OUTPUT_DIR, f"master_dataset_{WINDOW_SIZE}s.npz")
np.savez_compressed(
    out_file,
    X=X,
    y=y,
    s=s,
    file_id=file_id
)

del X, y, file_id
print("\n" + "=" * 60)
print(f"✅ Master dataset securely saved to {out_file}!")
print("=" * 60)
########################################################################################################
import numpy as np
import handy.my_utils as deniz
from EEGPreprocessor import EEGPreprocessor
import os
import gc
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
CHB_MIT_PATH = 'data-understanding/data/chb-mit'
NPZ_OUTPUT_DIR = "processed_master_datasets"
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

# ---> CHANGE THIS VALUE to generate datasets for 0.5, 1.0, 2.0, 4.0, 5.0, 10.0
WINDOW_SIZE = 10.0

# RULE 1: Zero Overlap. Windows must be strictly contiguous.
OVERLAP = 0.0

# RULE 2: Zero Padding reference. All EDF files must pad up to a multiple of 10s.
LARGEST_WINDOW_REF = 10.0

preprocessor = EEGPreprocessor(sampling_rate=256, target_sfreq=128)
new_fs = preprocessor.target_sfreq

WINDOW_SAMPLES = int(round(WINDOW_SIZE * new_fs))
STEP_SAMPLES = int(round(WINDOW_SAMPLES * (1 - OVERLAP)))

PATIENTS_TO_USE = [
    'chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
    'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
    'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
    'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24'
]

FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# =============================================================================
# SUMMARY PARSING
# =============================================================================
all_annotations = {}
patient_dirs = sorted(
    [d for d in os.listdir(CHB_MIT_PATH) if os.path.isdir(os.path.join(CHB_MIT_PATH, d)) and d.startswith('chb')])

print(f"\n👥 Parsing summary files for {len(patient_dirs)} patients...\n")
for patient in patient_dirs:
    summary_file = os.path.join(CHB_MIT_PATH, patient, f'{patient}-summary.txt')
    if os.path.exists(summary_file):
        seizure_info = deniz.parse_summary_file(summary_file)
        if seizure_info:
            all_annotations[patient] = seizure_info

exclude_list = ['chb12/chb12_29.edf', 'chb12/chb12_27.edf', 'chb12/chb12_28.edf']
for item in exclude_list:
    folder, filename = item.split('/')
    if folder in all_annotations and filename in all_annotations[folder]:
        del all_annotations[folder][filename]

# =============================================================================
# DATA EXTRACTION
# =============================================================================
# We will collect everything sequentially into these lists
master_X = []
master_y = []
master_s = []
master_file_id = []

print(f"\n🚀 Extracting strictly aligned {WINDOW_SIZE}s dataset...")
print("-" * 50)

for patient in PATIENTS_TO_USE:
    if patient not in all_annotations:
        continue

    patient_seizure_count = 0
    patient_normal_count = 0

    # Ensure files are processed in alphabetical order to maintain chronological integrity
    for filename in sorted(all_annotations[patient].keys()):
        seizure_times = all_annotations[patient][filename]
        file_path = os.path.join(CHB_MIT_PATH, patient, filename)

        if not os.path.exists(file_path):
            continue

        try:
            # 1. Load File
            signals_temp, ch_names, fs_temp = deniz.fix_eeg_channels_load_edf(
                file_path=file_path,
                final_channels=FINAL_CHANNELS,
                load_func=deniz.load_edf_file
            )
            if signals_temp is None:
                continue

            # 2. Preprocess (Resampling, filtering, etc.)
            signals_temp = preprocessor.preprocess(signals_temp)
            n_samples = signals_temp.shape[1]

            # -----------------------------------------------------------------
            # RULE 2: ZERO PADDING
            # -----------------------------------------------------------------
            duration_sec = n_samples / new_fs
            remainder = duration_sec % LARGEST_WINDOW_REF

            if remainder > 0:
                pad_sec = LARGEST_WINDOW_REF - remainder
                pad_samples = int(round(pad_sec * new_fs))

                padding = np.zeros((signals_temp.shape[0], pad_samples), dtype=np.float32)
                signals_temp = np.concatenate([signals_temp, padding], axis=1)

            n_samples_padded = signals_temp.shape[1]

            # 3. Extract Windows Sequentially
            for start in range(0, n_samples_padded - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                end = start + WINDOW_SAMPLES
                window = signals_temp[:, start:end]

                window_start_sec = start / new_fs
                window_end_sec = end / new_fs

                # -------------------------------------------------------------
                # ENSEMBLE ALIGNMENT RULE: NEVER DROP WINDOWS
                # -------------------------------------------------------------
                overlap_duration = 0.0
                for sz_start, sz_end in seizure_times:
                    overlap_start = max(window_start_sec, sz_start)
                    overlap_end = min(window_end_sec, sz_end)

                    if overlap_end > overlap_start:
                        overlap_duration += (overlap_end - overlap_start)

                overlap_ratio = overlap_duration / WINDOW_SIZE

                # Kesin Karar: %50'dan büyük eşitse 1, değilse 0. Hiçbir pencere çöpe atılmaz.
                is_seizure = (overlap_ratio >= 0.50)

                master_X.append(window.astype(np.float32))
                master_y.append(1 if is_seizure else 0)
                master_s.append(patient)
                master_file_id.append(filename)

                if is_seizure:
                    patient_seizure_count += 1
                else:
                    patient_normal_count += 1

            # Clear memory
            del signals_temp
            gc.collect()

        except Exception as e:
            print(f"   [!] Error processing {filename}: {e}")
            continue

    print(f"   ✓ {patient}: {patient_seizure_count} seizure, {patient_normal_count} normal windows (100% Retained)")
    gc.collect()

# =============================================================================
# SAVING THE DATASET
# =============================================================================
print("\n" + "-" * 50)
print("📦 Compiling and saving the master dataset...")

# Initial shape will be (N, Channels, Window_Samples)
X = np.array(master_X, dtype=np.float32)
y = np.array(master_y, dtype=np.int8)
s = np.array(master_s, dtype=str)
file_id = np.array(master_file_id, dtype=str)

# Clear lists to free RAM before transposing and saving
del master_X, master_y, master_s, master_file_id
gc.collect()

# -----------------------------------------------------------------------------
# TRANSPOSING THE DATA
# -----------------------------------------------------------------------------
X = np.transpose(X, (2, 0, 1))

print(f"\n📈 Final {WINDOW_SIZE}s Dataset:")
print(f"   • Total Windows: {len(y)}")
print(f"   • Shape of X:    {X.shape} (Window Size, Memory Size, Channels)")
print(f"   • Memory Size:   ~{X.nbytes / 1024 / 1024 / 1024:.2f} GB")

out_file = os.path.join(NPZ_OUTPUT_DIR, f"master_dataset_{WINDOW_SIZE}s.npz")
np.savez_compressed(
    out_file,
    X=X,
    y=y,
    s=s,
    file_id=file_id
)

del X, y, file_id
print("\n" + "=" * 60)
print(f"✅ Master dataset securely saved to {out_file}!")
print("=" * 60)