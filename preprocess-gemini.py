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
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir if split == "train" else val_dir, exist_ok=True)

    for edf_path in edf_list:
        file_name = os.path.basename(edf_path)
        subject = file_name.split("_")[0]
        print(f"\nProcessing: {file_name} [{split}]")

        try:
            # ADIM 1: Raw Veriyi Yükle
            # fix_eeg_channels fonksiyonunuzun ham veriyi (Raw objesi) döndürdüğünü varsayıyorum.
            # ÖNEMLİ: fix_eeg_channels içinde zamanı kırpan (crop) bir işlem olmadığından emin olun.
            raw = deniz.fix_eeg_channels_version_2(edf_path, final_channels)
            if raw is None: continue

            # ADIM 2: Anotasyonları HEMEN Ekle (İşlemlerden Önce!)
            # Bu fonksiyon CSV'den saniyeleri okuyup MNE Annotation objesi yapıyor
            annotations, _, _ = deniz.build_seizure_annotations_for_file_v2(seizure_df, file_name)

            # Eğer dosya başlangıcı (meas_date) varsa, MNE saniyeleri tarihle karıştırabilir.
            # CHB-MIT'de basit saniye takibi için orig_time=None tutmak genelde daha güvenlidir.
            if annotations:
                raw.set_annotations(annotations)

            # ADIM 3: Etiket Maskesini Oluştur (Downsample öncesi)
            # Böylece etiketler de sinyalle birlikte downsample edilebilir veya
            # downsample sonrası tekrar hesaplanabilir.
            # En temiz yöntem: İşlemleri yap, en son maskeyi oluştur.
            # Ancak anotasyonları raw'a eklediğimiz için MNE onları taşıyacak.

            # --- Preprocess ---
            raw = deniz.downsample_version2(raw)
            raw = deniz.apply_filtering_version2(raw)

            # ADIM 4: Maskeyi Şimdi Oluştur (Sample rate oturduktan sonra)
            # Raw üzerinde annotationlar var, şimdi sample indexlere çeviriyoruz.
            label_mask = deniz.create_label_mask(raw, raw.annotations,
                                           pre_exclude=preictal_exclude,
                                           post_exclude=postictal_exclude)

            # ADIM 5: Epoch'lama (Hem Data Hem Label İçin)
            # Data Epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, overlap=overlap, preload=True,
                                                  verbose=False)
            X = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)

            # Label Epochs
            # Maskeyi de aynı şekilde bölmek için onu geçici bir Raw objesine çevirebilir
            # ya da reshape yapabiliriz. Reshape en hızlısıdır.
            # Önemli: MNE make_fixed_length_epochs bazen sondaki tam olmayan veriyi atar.
            # Bu yüzden epochs.events'i referans almalıyız.

            y_epoch_list = []
            n_samples_per_epoch = X.shape[2]  # Örn: 256 * 2 = 512

            for event in epochs.events:
                start_samp = event[0]
                end_samp = start_samp + n_samples_per_epoch

                # Maskeden ilgili parçayı kes
                mask_chunk = label_mask[start_samp:end_samp]

                # Karar Mantığı (Voting):
                # Eğer chunk içinde '1' (seizure) varsa, epoch = Seizure (1)
                # Eğer chunk tamamen -1 ise, epoch = Exclude (-1)
                # Karışık durumlarda öncelik sırası: Seizure > Exclude > Safe

                if 1 in mask_chunk:
                    label = 1
                elif -1 in mask_chunk:  # Sadece exclude ve safe varsa
                    # Eğer epoch'un yarısından fazlası exclude ise atalım
                    if np.sum(mask_chunk == -1) > (n_samples_per_epoch * 0.5):
                        label = -1
                    else:
                        label = 0
                else:
                    label = 0

                y_epoch_list.append(label)

            y = np.array(y_epoch_list)

            # ADIM 6: Temizlik ve Kayıt
            # Exclude (-1) olanları at
            valid_mask = (y != -1)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            if len(y_clean) == 0:
                print(f"  ⚠️ Dosya tamamen exclude edildi: {file_name}")
                continue

            # --- Sampling Stratejisi (Düzeltilmiş) ---
            if split == "train":
                seizure_idx = np.where(y_clean == 1)[0]
                safe_idx = np.where(y_clean == 0)[0]

                if len(seizure_idx) > 0:
                    # ÖNERİ: 1:1 yerine 1:10 gibi bir oran kullanın veya
                    # nöbet yoksa dosyayı tamamen atmak yerine background verisi alın.
                    # Mevcut kodunuzdaki mantığı koruyup biraz gevşetiyorum:
                    n_seizure = len(seizure_idx)
                    # Nöbetin 5-10 katı kadar normal veri al (Modeli kör etmemek için)
                    n_take = min(len(safe_idx), n_seizure)

                    # Rastgele seçim (Shuffle edip baştan al)
                    safe_chosen = np.random.choice(safe_idx, size=n_take, replace=False)

                    final_idx = np.concatenate([seizure_idx, safe_chosen])
                    X_final, y_final = shuffle(X_clean[final_idx], y_clean[final_idx], random_state=42)
                else:
                    # Nöbet yoksa, yine de veri almalıyız (False Positive eğitimi için)
                    # Ama çok değil, örn 200 epoch.
                    n_take = min(200, len(safe_idx))
                    if n_take > 0:
                        chosen_idx = np.random.choice(safe_idx, size=n_take, replace=False)
                        X_final, y_final = X_clean[chosen_idx], y_clean[chosen_idx]
                    else:
                        continue
            else:
                X_final, y_final = X_clean, y_clean

            # Kaydet
            target_dir = train_dir if split == "train" else val_dir
            subject_dir = os.path.join(target_dir, subject)
            os.makedirs(subject_dir, exist_ok=True)

            save_path = os.path.join(subject_dir, file_name.replace(".edf", f"_{split}.npz"))
            np.savez_compressed(save_path, X=X_final, y=y_final)

            print(f"  → Saved: {X_final.shape}, Seizure Count: {np.sum(y_final == 1)}")

            del raw, epochs, X, label_mask
            gc.collect()

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()



process_and_save_corrected(
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

process_and_save_corrected(
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
# path = "dataset_final_gemini/train/chb01/chb01_03_train.npz"
# npz = np.load(path)
# npz['X']
# np.count_nonzero(npz['y'])
# len(npz['y'])
