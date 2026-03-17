import os
import numpy as np
import pandas as pd


def window_level_align(coarse_probs: np.ndarray, target_length: int) -> np.ndarray:
    n_coarse = len(coarse_probs)
    if n_coarse == target_length:
        return coarse_probs.copy()
    fine_indices = np.floor(np.linspace(0, n_coarse, target_length, endpoint=False)).astype(int)
    fine_indices = np.clip(fine_indices, 0, n_coarse - 1)
    return coarse_probs[fine_indices]


def analyze_all_resolutions():
    # Veri setlerinin yolları
    dataset_paths = {
        '0.5s': 'master_dataset_0.5s.npz',
        '1s': 'master_dataset_1s.npz',  # REFERANS
        '2s': 'master_dataset_2s.npz',
        '4s': 'master_dataset_4s.npz',
        '5s': 'master_dataset_5s.npz',
        '10s': 'master_dataset_10s.npz'
    }

    # 1. Referans (1s) verisini yükle ve hastayı seç
    if not os.path.exists(dataset_paths['1s']):
        print(f"Referans veri seti bulunamadı: {dataset_paths['1s']}")
        return

    print("Referans 1s verisi yükleniyor...")
    data_1s = np.load(dataset_paths['1s'])
    subjects = np.unique(data_1s['s'])
    test_subject = str(subjects[0])  # İlk hastayı seç

    mask_1s = data_1s['s'] == test_subject
    y_1s_ref = data_1s['y'][mask_1s]
    target_len = len(y_1s_ref)

    print(f"\nAnaliz edilecek hasta: {test_subject}")
    print(f"Hedef (1s) uzunluk: {target_len} adım\n")

    # 2. Tüm etiketleri toplayacağımız sözlük
    aligned_labels = {
        'Index (1s Bazlı)': np.arange(target_len),
        '1s (Referans)': y_1s_ref.astype(int)
    }

    # 3. Diğer veri setlerini döngüyle yükle ve hizala
    for res, path in dataset_paths.items():
        if res == '1s':
            continue  # Referansı zaten aldık

        if not os.path.exists(path):
            print(f"UYARI: {path} bulunamadı, atlanıyor...")
            continue

        print(f"{res} verisi yükleniyor ve hizalanıyor...")
        data = np.load(path)
        mask = data['s'] == test_subject
        y_raw = data['y'][mask]

        # 1s uzunluğuna hizala ve temiz görünüm için integer'a çevir
        y_aligned = window_level_align(y_raw, target_len)
        aligned_labels[f'{res} Hizalanmış'] = y_aligned.astype(int)

    # 4. DataFrame oluştur
    df = pd.DataFrame(aligned_labels)

    # 5. Sadece kriz anlarının olduğu kritik bir bölgeyi bul ve yazdır
    seizure_indices = df[df['1s (Referans)'] == 1].index

    if len(seizure_indices) > 0:
        start_idx = max(0, seizure_indices[0] - 5)  # Krizden 5 saniye öncesi
        end_idx = min(len(df), seizure_indices[0] + 35)  # Kriz anı ve sonrası

        print("\n" + "=" * 80)
        print(" TÜM ÇÖZÜNÜRLÜKLERDE ETİKET (LABEL) HİZALAMA ANALİZİ")
        print("=" * 80)
        print(df.iloc[start_idx:end_idx].to_string(index=False))
        print("=" * 80)
    else:
        print("Bu hastada kriz (y=1) bulunamadı.")


if __name__ == '__main__':
    analyze_all_resolutions()