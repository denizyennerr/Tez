import os
import numpy as np
from collections import defaultdict

DATA_DIR = "processed_npz_files"

all_seizure_windows_dict = defaultdict(list)
all_normal_windows_dict = defaultdict(list)

npz_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]

print(f"📁 Toplam NPZ dosyası: {len(npz_files)}")

for file in npz_files:
    subject = file.split("_")[0]  # chb01_chb01_03.npz → chb01
    path = os.path.join(DATA_DIR, file)

    data = np.load(path)
    X = data["X"]  # (samples, channels, time)
    y = data["y"]

    for i in range(len(y)):
        if y[i] == 1:
            all_seizure_windows_dict[subject].append(X[i])
        else:
            all_normal_windows_dict[subject].append(X[i])

print("✅ Dictionary oluşturuldu.\n")

# Subject başına istatistik
for sub in sorted(all_seizure_windows_dict.keys()):
    print(f"{sub} → Seizure: {len(all_seizure_windows_dict[sub])}, "
          f"Normal: {len(all_normal_windows_dict[sub])}")

all_subjects = sorted(list(all_seizure_windows_dict.keys()))
print("👥 Subjects:", all_subjects)

### anlatım
np.random.seed(42)
np.random.shuffle(all_subjects)

train_ratio = 0.7
val_ratio = 0.15

n_total = len(all_subjects)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_subjects = all_subjects[:n_train]
val_subjects = all_subjects[n_train:n_train + n_val]
test_subjects = all_subjects[n_train + n_val:]
### anlatım


# Bu fonksiyonu çalıştırmak için verilerini hasta bazlı bir dict yapısında toplaman gerekir.


# ============================================
# CELL 12: Subject-Level Train/Val/Test Split
# ============================================

import numpy as np
import gc
from sklearn.model_selection import train_test_split

print("=" * 60)
print("✂️ SUBJECT-LEVEL DATA SPLITTING (No Leakage)")
print("=" * 60)

# 1. Hastaları Manuel Olarak Gruplandır (Subject-wise)
# Akademik olarak en sağlam yöntem: Test setindeki hastayı eğitimde hiç göstermemektir.
train_subjects = ['chb24', 'chb15', 'chb14', 'chb10', 'chb16']
val_subjects = ['chb01']
test_subjects = ['chb03', 'chb06']

set(train_subjects) & set(val_subjects)
set(train_subjects) & set(test_subjects)
set(val_subjects) & set(test_subjects)
#######
def get_data_by_subjects(subject_list, all_seizure_windows_dict, all_normal_windows_dict):
    X_list, y_list = [], []
    for sub in subject_list:
        if sub in all_seizure_windows_dict:
            # Seizure pencerelerini ekle
            X_list.append(np.array(all_seizure_windows_dict[sub]))
            y_list.append(np.ones(len(all_seizure_windows_dict[sub])))

            # Normal pencereleri ekle
            X_list.append(np.array(all_normal_windows_dict[sub]))
            y_list.append(np.zeros(len(all_normal_windows_dict[sub])))

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


X_train_raw, y_train_raw = get_data_by_subjects(
    train_subjects,
    all_seizure_windows_dict,
    all_normal_windows_dict
)

X_val_raw, y_val_raw = get_data_by_subjects(
    val_subjects,
    all_seizure_windows_dict,
    all_normal_windows_dict
)

X_test_raw, y_test_raw = get_data_by_subjects(
    test_subjects,
    all_seizure_windows_dict,
    all_normal_windows_dict
)

# NOT: Bu adımın çalışması için CELL 11'de pencereleri toplarken
# 'all_seizure_windows' yerine 'all_seizure_windows_dict[patient]'
# şeklinde sözlük yapısında saklamış olmanız gerekir.

# Varsayımsal olarak verileri ayırıyoruz:
print(f"👥 Training Subjects: {train_subjects}")
print(f"👥 Validation Subject: {val_subjects}")
print(f"👥 Test Subject:      {test_subjects}")



def prepare_set(X_raw, y_raw):
    # (samples, channels, time) -> (samples, time, channels)
    X_res = np.transpose(X_raw, (0, 2, 1))
    # Karıştır (Sadece kendi seti içinde!)
    idx = np.random.permutation(len(X_res))
    return X_res[idx], y_raw[idx]


# Bu kısımlar CELL 11'den gelen sözlük yapısına göredir:
X_train, y_train = prepare_set(X_train_raw, y_train_raw)
X_val, y_val = prepare_set(X_val_raw, y_val_raw)
X_test, y_test = prepare_set(X_test_raw, y_test_raw)

np.savez('train_set.npz', X_train=X_train, y_train=y_train)
np.savez('val_set.npz', X_val=X_val, y_val=y_val)
np.savez('test_set.npz', X_test=X_test, y_test=y_test)

print(f"\n📊 Final Split Results (Subject-Wise):")
print(f"   ┌─────────────┬─────────┬─────────┬─────────┐")
print(f"   │ Set         │ Samples │ Seizure │ Normal  │")
print(f"   ├─────────────┼─────────┼─────────┼─────────┤")
print(f"   │ Train       │ {len(X_train):>7} │ {int(np.sum(y_train)):>7} │ {int(len(y_train) - np.sum(y_train)):>7} │")
print(f"   │ Validation  │ {len(X_val):>7} │ {int(np.sum(y_val)):>7} │ {int(len(y_val) - np.sum(y_val)):>7} │")
print(f"   │ Test        │ {len(X_test):>7} │ {int(np.sum(y_test)):>7} │ {int(len(y_test) - np.sum(y_test)):>7} │")
print(f"   └─────────────┴─────────┴─────────┴─────────┘")

gc.collect()
print("\n" + "=" * 60)
print("✅ Subject-level split complete! Ready for Thesis.")
print("=" * 60)