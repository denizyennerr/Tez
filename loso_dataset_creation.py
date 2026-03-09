import os
import numpy as np
from collections import defaultdict
import gc

DATA_DIR = "processed_npz_files_10s"
save_as= 'master_dataset_10s.npz'

all_subjects_data = []
all_labels = []
subject_mapping = []  # Her pencerenin hangi hastaya ait olduğunu tutacak

npz_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])

print(f"📁 Toplam {len(npz_files)} dosya işleniyor...")

for file in npz_files:
    subject_id = file.split("_")[0]  # Örn: 'chb01'
    path = os.path.join(DATA_DIR, file)

    data = np.load(path)
    X = data["X"]  # (samples, channels, time)
    y = data["y"]

    # Eksen kaydırma: (samples, channels, time) -> (samples, time, channels)
    X = np.transpose(X, (0, 2, 1))

    all_subjects_data.append(X)
    all_labels.append(y)
    # Her örnek için hangi subject olduğunu kaydet (sayısal veya string)
    subject_mapping.extend([subject_id] * len(y))

# Listeleri devasa numpy dizilerine dönüştür
X_all = np.concatenate(all_subjects_data, axis=0)
y_all = np.concatenate(all_labels, axis=0)
s_all = np.array(subject_mapping)

# Belleği rahatlat
del all_subjects_data, all_labels, subject_mapping
gc.collect()

# Master dosyayı kaydet
np.savez_compressed(save_as, X=X_all, y=y_all, s=s_all)

print(f"\n✅ Master Dataset Oluşturuldu!")
print(f"📊 Toplam Örnek: {len(X_all)}")
print(f"👥 Benzersiz Denekler: {np.unique(s_all)}")
print(f"💾 Dosya: {save_as}")