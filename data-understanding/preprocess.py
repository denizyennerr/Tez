from utility.my_utils import deniz
import os
import warnings
import gc
import mne
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

dataset_path = 'data-understanding/data/chb-mit'
subject_folders = deniz.get_folder_names(dataset_path)
edf_paths = deniz.get_edf_paths(dataset_path, subject_folders)
FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]
output_folders = ['2s_epochs_processed', '4s_epochs_processed', '6s_epochs_processed', '8s_epochs_processed',
                  '10s_epochs_processed']
epoch_lengths = [2, 4, 6, 8, 10]

df = pd.read_csv('data-understanding/only_seizures_for_preprocess.csv')
df['file'] = df['file'].str.replace('_processed', '', regex=False)
# df.to_csv('data-understanding/all_preprocess_pipeline_seizure.csv', index=False)




def preprocess_pipe(seizure_paths, final_channels, df, epoch_length, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📂 '{output_folder}' klasörü oluşturuldu.")

    processed_count = 0

    for file_path in seizure_paths:
        file_name_full = os.path.basename(file_path)
        file_name_base, _ = os.path.splitext(file_name_full)

        # 1. Ham Veriyi Oku ve Kanalları Düzenle
        raw = deniz.fix_eeg_channels(file_path, final_channels)

        if raw is not None:
            try:
                # 2. Ön İşleme Adımları
                raw = deniz.downsample(raw)
                raw = deniz.apply_filtering(raw)

                # 3. Annotasyon Ekleme
                annotations, _, _ = deniz.build_seizure_annotations_for_file(df=df, file_name=file_name_full)

                if annotations is not None and len(annotations) > 0:
                    raw.set_annotations(annotations)
                    print(f"✅ Annotasyonlar eklendi: {file_name_full}")
                else:
                    print(f"ℹ️ Annotasyon bulunamadı: {file_name_full}")

                # 4. Epochs Oluşturma
                epochs = mne.make_fixed_length_epochs(raw, preload=True, verbose=False, duration=epoch_length)

                # 5. Her Epoch İçin Label Üretimi
                labels = deniz.generate_epoch_labels(epochs, raw)

                # Epoch verisini al (numpy array formatında)
                # Shape: (n_epochs, n_channels, n_times)
                data = epochs.get_data()

                print(f"💾 {file_name_full} için {len(epochs)} adet .npz dosyası kaydediliyor...")

                # 6. Her Epoch'u ve Label'ı Ayrı Kaydetme
                for i in range(len(epochs)):
                    # X: (channels, times), y: scalar (0 veya 1)
                    X_epoch = data[i]
                    y_epoch = labels[i]

                    # Dosya ismi: chb01_03_E005_L1.npz (L: Label bilgisini isme de ekleyelim)
                    epoch_file_name = f"{file_name_base}_E{i:03d}_L{y_epoch}.npz"
                    save_path = os.path.join(output_folder, epoch_file_name)

                    # Veriyi ve etiketi sıkıştırılmış numpy dosyası olarak kaydet
                    np.savez_compressed(save_path, x=X_epoch, y=y_epoch)

                processed_count += 1

            except Exception as e:
                print(f"❌ İşleme hatası ({file_name_full}): {e}")

            finally:
                # Belleği temiz tut
                if 'raw' in locals(): del raw
                if 'epochs' in locals(): del epochs
                if 'data' in locals(): del data
                gc.collect()
        else:
            print(f"⏭️ {file_name_full} atlandı.")

    print(f"\n✨ İşlem bitti! Toplam {processed_count} dosya işlendi.")


preprocess_pipe(seizure_paths=edf_paths[2:3], final_channels=FINAL_CHANNELS, df=df, epoch_length=2,
                output_folder='02_epochs')