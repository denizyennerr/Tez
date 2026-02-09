import importlib

import numpy as np
import os
import gc
import mne
import pandas as pd
from utility.my_utils import deniz
import random
from pathlib import Path
importlib.reload(deniz)

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
len(set(df['file']))  # 138
len(df['file'])  # 185


# 0. İstenmeyen dosyaların listesi
exclude_list = [
    'chb12/chb12_29.edf',
    'chb12/chb12_27.edf',
    'chb12/chb12_28.edf'
]

# 1. Seizure içeren dosyaların listesini al (Unique)
seizure_file_names = set(df['file'].unique())

# 2. Tüm yollar içerisinden ayırma ve filtreleme
seizure_paths = []
non_seizure_paths = []

for path in edf_paths:
    # Dosya adını al (örn: chb01_01.edf)
    f_name = os.path.basename(path)
    # Klasör adını al (örn: chb01)
    f_folder = os.path.basename(os.path.dirname(path))
    # Eşleştirme formatı oluştur (örn: chb01/chb01_01.edf)
    relative_path = f"{f_folder}/{f_name}"

    # EĞER dosya exclude_list içindeyse direkt atla
    if relative_path in exclude_list:
        print(f"🚫 Filtrelendi (Exclude): {relative_path}")
        continue

    # Değilse seizure durumuna göre listelere ekle
    if f_name in seizure_file_names:
        seizure_paths.append(path)
    else:
        non_seizure_paths.append(path)

# 3. Eşitlik sağlamak için non_seizure listesinden rastgele seçim yap
random.seed(42)  # Tekrarlanabilirlik için

# Kaç tane seizure dosyası varsa o kadar sağlıklı dosya seçiyoruz
num_samples = len(seizure_paths)
selected_non_seizure_paths = random.sample(non_seizure_paths, num_samples)

# 4. Final listesini birleştir
final_path_list = seizure_paths + selected_non_seizure_paths

print("-" * 30)
print(f"✅ Toplam Seizure Dosyası: {len(seizure_paths)}")
print(f"✅ Seçilen Sağlıklı Dosya: {len(selected_non_seizure_paths)}")
print(f"🚀 İşlem Listesi Hazır: Toplam {len(final_path_list)} dosya.")

###############################################


def preprocess_pipe(file_paths, final_channels, df, epoch_length, output_folder):
    # Klasörü oluştur ve tam yolunu (absolute path) al
    output_path = Path(output_folder).resolve()
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📂 '{output_path}' klasörü oluşturuldu.")

    chunk_duration = 3600  # 1 Saat
    processed_count = 0

    for file_path in file_paths:
        file_name_full = os.path.basename(file_path)
        file_name_base, _ = os.path.splitext(file_name_full)

        # 1. Ham veriyi oku
        raw_original = deniz.fix_eeg_channels_version_2(file_path, final_channels)

        if raw_original is not None:
            total_duration = raw_original.times[-1]
            num_chunks = int(total_duration // chunk_duration)

            if num_chunks == 0: num_chunks = 1  # 1 saatten kısa dosyalar için

            for i in range(num_chunks):
                try:
                    tmin = i * chunk_duration
                    tmax = min((i + 1) * chunk_duration, total_duration)  # Son parçada taşma olmasın

                    if (tmax - tmin) < chunk_duration and i > 0:
                        # Eğer 1 saatten kısa bir artan kısımsa ve ilk parça değilse atla
                        break

                    # PARÇAYI KOPYALA VE KES
                    raw = raw_original.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)

                    # 2. Ön işleme (Resample ve Filtre)
                    raw = deniz.downsample_version2(raw)
                    raw = deniz.apply_filtering_version2(raw)

                    # 3. Annotasyonları ekle
                    annotations, _, _ = deniz.build_seizure_annotations_for_file(df=df, file_name=file_name_full)
                    if annotations is not None:
                        # MNE burada 'Omitted' uyarısı verebilir, normaldir.
                        raw.set_annotations(annotations)

                    # 4. Epochlara böl
                    epochs = mne.make_fixed_length_epochs(raw, preload=True, verbose=False, duration=epoch_length)

                    # 5. Label Üretimi
                    labels = deniz.generate_epoch_labels(epochs, raw)

                    data = epochs.get_data()
                    labels = np.array(labels)

                    # 6. Kayıt (Path objesi kullanarak güvenli kaydetme)
                    save_name = f"{file_name_base}_P{i + 1}.npz"
                    final_save_path = output_path / save_name

                    np.savez_compressed(str(final_save_path), x=data, y=labels)

                    print(f"✅ Kaydedildi: {save_name} | SFREQ: {raw.info['sfreq']} | Shape: {data.shape}")
                    processed_count += 1

                except Exception as e:
                    print(f"❌ Hata: {file_name_full} Part {i + 1}: {e}")
                finally:
                    if 'raw' in locals(): del raw
                    gc.collect()

            del raw_original
            gc.collect()
        else:
            print(f"⏭️ Atlandı: {file_name_full}")

    print(f"\n✨ İşlem bitti! Toplam {processed_count} .npz dosyası '{output_path}' konumuna kaydedildi.")


preprocess_pipe(file_paths=final_path_list, final_channels=FINAL_CHANNELS,df=df, epoch_length=2, output_folder='processed_dataset_2')



