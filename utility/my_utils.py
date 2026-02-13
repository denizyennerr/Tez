import gc

import mne
import numpy as np
import pandas as pd
import os
from pathlib import Path
import unicodedata
import re
import matplotlib.pyplot as plt
import json
import shutil
import numpy as np
import os
import random


TARGET_SFREQ = 128.0  # Hz
SOURCE_SFREQ = 256.0  # Hz


##Inspecting data
def get_folder_names(folder_path: str):
    """ Returns a list of all files in folder_path"""
    try:
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        return (sorted(folders))
    except FileNotFoundError:
        return "Path not found, please check folder_path"


def get_edf_paths(dataset_path, subject_folders):
    """
    Gets all the .edf files under certain directory.
    Belirli klasörler altındaki tüm .edf dosyalarının tam yollarını liste olarak döner.
    subject_folders: Tek bir string (klasör adı) veya klasör adlarından oluşan bir liste olabilir.
    Example:
    all_edf_files = get_edf_paths(dataset_path, every_subject_folder)
    chb01_files = get_edf_paths(dataset_path, every_subject_folder[0])
    """
    edf_paths = []

    # If only one directory is used (string), convert it to list
    if isinstance(subject_folders, str):
        subject_folders = [subject_folders]

    for folder in subject_folders:
        folder_full_path = os.path.join(dataset_path, folder)

        # Check whether path exists
        if os.path.exists(folder_full_path):
            # Scan the files inside the path
            for file in os.listdir(folder_full_path):
                if file.endswith(".edf"):
                    # Create the full file path and append it to all edf paths
                    full_file_path = os.path.join(folder_full_path, file)
                    edf_paths.append(full_file_path)
        else:
            print(f"Uyarı: {folder_full_path} dizini bulunamadı.")

    return sorted(edf_paths)


def copy_files_to_seizured_folder(file_list, target_folder="seizured_files"):
    """
    Listedeki dosyaları belirtilen hedef klasöre kopyalar.
    Klasör yoksa oluşturur.

    # Örnek Kullanım:
    # somelist = ["data-understanding/data/chb-mit/chb01_01.edf", ...]
    # copy_files_to_seizured_folder(somelist)
    """
    # 1. Hedef klasörü oluştur (varsa hata vermez)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Klasör oluşturuldu: {target_folder}")
    else:
        print(f"Klasör zaten mevcut: {target_folder}")

    copied_count = 0
    error_count = 0

    # 2. Dosyaları kopyala
    for file_path in file_list:
        try:
            # Sadece dosya adını al (yolun tamamını değil)
            file_name = os.path.basename(file_path)
            destination = os.path.join(target_folder, file_name)

            # copy2: İçerik + Meta verileri (mtime, atime vb.) kopyalar
            shutil.copy2(file_path, destination)
            copied_count += 1

        except FileNotFoundError:
            print(f"Hata: Dosya bulunamadı -> {file_path}")
            error_count += 1
        except Exception as e:
            print(f"Beklenmeyen hata ({file_path}): {e}")
            error_count += 1

    print(f"\nİşlem tamamlandı!")
    print(f"Başarıyla kopyalanan: {copied_count}")
    print(f"Hata sayısı: {error_count}")


def channel_len_check(edf_path, verbose=False, x=0):
    '''
    for each edf file checks the lengths of the channels. If all of them is identical, they are checked and returned.
    If verbose is true prints small report.
    :param edf_path: list of all files in the chb folder
    :param verbose1: boolean
    :param verbose: boolean
    :param x: integer aribatry edf_path index choosing
    :return: integer: number of channels
    ---
    usage
    channel_len_check(edf_path_1)
    or
    variable = channel_len_check(edf_path_1)
    '''
    raw = mne.io.read_raw_edf(edf_path[x], preload=False, verbose=False)
    counter = 0
    len_channel = len(raw.ch_names)
    if verbose:
        print('this many files inside of the path: ', len(edf_path))
        print("--")
        print("first edfs channel len is", len_channel)

    for i in edf_path:
        raw = mne.io.read_raw_edf(i, verbose=False, preload=False)
        if len(raw.ch_names) == len_channel:
            counter += 1
        else:
            print("in this subject every edf's channel len is not same: ", i)
            print("******")

    if counter == len(edf_path):
        print("in this subject every edf's channel len is", len_channel)
        return len_channel


def get_channels_as_dict(dataset_path, subject_folders, json_name: str, save=False):
    '''
    takes every edf files channels inside a dict and returns and as variable and saves as json file.
    Also it is avaliable to take action only for one folder.

    :param dataset_path:
    :param subject_folders:
    :param json_name: str -> json file name
    :return: channel_dict
    -----------
    usage:
    channel_dict = get_channels_as_dict(dataset_path,subject_folders)
    channel_dict = get_channels_as_dict(dataset_path,subject_folders[0])
    '''

    if isinstance(subject_folders, str):
        subject_folders = [subject_folders]

    edf_paths = get_edf_paths(dataset_path, subject_folders)
    channel_dict = {}
    for path in edf_paths:
        channel_dict[path] = mne.io.read_raw_edf(path, verbose=False).ch_names

    # channel_dict save
    if len(channel_dict) > 0 & save:
        with open(f"{json_name}.json", "w", encoding="utf-8") as f:
            json.dump(channel_dict, f, sort_keys=False, indent=4, ensure_ascii=False)

    return channel_dict


def check_is_every_edf_containts_target_chs(edf_paths, target_set, missing_files):
    for path in edf_paths:
        try:
            # Sadece header'ı oku (preload=False zaten hızlıdır)
            raw = mne.io.read_raw_edf(path, verbose=False, preload=False)
            current_channels = raw.ch_names

            # Subset kontrolü: target_set içindeki her bir kanal mevcut mu?
            # Not: CHB-MIT'de bazen 'FP1-F7' yerine 'T8-P8-1' gibi isimler olabilir.
            # Eğer tam eşleşme istiyorsan:
            is_subset = target_set.issubset(set(current_channels))

            if is_subset:
                print(f"✅ Uygun: {path.split('/')[-1]}")
            else:
                missing = target_set - set(current_channels)
                print(f"❌ Eksik kanal var: {path.split('/')[-1]} -> Eksikler: {missing}")
                missing_files.append(path)

        except Exception as e:
            print(f"⚠️ Dosya okunurken hata oluştu: {path} - Hata: {e}")


def normalize_channel_name(ch: str) -> str:
    """
    EEG channel isimlerini deterministik ve güvenli şekilde normalize eder.
    """

    # 1. Unicode normalize (gizli karakterleri temizler)
    ch = unicodedata.normalize("NFKD", ch)

    # 2. Tüm tire varyantlarını ASCII '-' yap
    ch = ch.replace("–", "-").replace("—", "-").replace("−", "-")

    # 3. Baştaki / sondaki boşlukları sil
    ch = ch.strip()

    # 4. İçteki fazla boşlukları teke indir
    ch = re.sub(r"\s+", " ", ch)

    # 5. Büyük harfe çevir
    ch = ch.upper()

    return ch


def normalize_channel_dict(channel_dict: dict) -> dict:
    """
    {path: [channels]} sözlüğündeki tüm channel isimlerini normalize eder.
    """

    normalized = {}

    for path, channels in channel_dict.items():
        normalized[path] = [
            normalize_channel_name(ch)
            for ch in channels
        ]

    return normalized


def channel_checking_rolling(dataset_path, verbose=False):
    '''
    :param dataset_path:
    :param verbose:
    :return:
    '''
    every_subject_folder = get_folder_names(dataset_path)

    for i in every_subject_folder:
        edf_path = get_edf_paths(dataset_path, i)
        first_channel_len = channel_len_check(edf_path, verbose=verbose)
        print(edf_path)
        print("#########")


def diagnose_channel_names(dataset_path, folder_name, num_files=5):
    """
    Inspect channel names across multiple files to identify naming patterns
    """
    print("=" * 70)
    print("CHANNEL NAMING DIAGNOSTIC FOR CHB-MIT DATASET")
    print("=" * 70)

    all_unique_channels = set()
    channel_variations = {}

    for i in range(min(num_files, len(folder_name))):
        current_folder = folder_name[i]

        # Construct file path
        if 'chb08' in current_folder:
            filename = f"{current_folder}_02.edf"
        else:
            filename = f"{current_folder}_01.edf"

        single_edf = os.path.join(dataset_path, current_folder, filename)

        print(f"\n[{i + 1}] File: {filename}")
        print("-" * 70)

        try:
            raw = mne.io.read_raw_edf(single_edf, preload=False, verbose='error')

            print(f"Total channels: {len(raw.ch_names)}")
            print(f"Channel names (raw):")
            for idx, ch in enumerate(raw.ch_names, 1):
                # Show original name and what it would become after cleaning
                clean_name = ch.upper().strip().rstrip('.')
                if clean_name.endswith('-0') or clean_name.endswith('-1') or clean_name.endswith('-2'):
                    clean_name = clean_name[:-2]

                marker = " ← NEEDS CLEANING" if ch != clean_name else ""
                print(f"  {idx:2d}. '{ch}' → '{clean_name}'{marker}")

                all_unique_channels.add(clean_name)

                # Track variations
                if clean_name not in channel_variations:
                    channel_variations[clean_name] = set()
                channel_variations[clean_name].add(ch)

        except Exception as e:
            print(f"ERROR reading file: {e}")


def standardize_channels(raw):
    """
    Renames, selects, and reorders channels to match TARGET_CHANNELS.

    Handles common CHB-MIT naming variations:
    - Removes trailing dots (e.g., 'FZ-CZ.' -> 'FZ-CZ')
    - Removes suffixes like -0, -1 (e.g., 'T8-P8-1' -> 'T8-P8')
    - Converts to uppercase for consistency

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data object

    Returns:
    --------
    mne.io.Raw or None
        Standardized raw object with TARGET_CHANNELS, or None if missing channels
    """
    # Step 1: Build rename mapping
    rename_dict = {}
    for name in raw.ch_names:
        # Normalize: Uppercase -> Remove dots -> Remove numeric suffixes
        clean_name = name.upper().strip()
        clean_name = clean_name.rstrip('.')  # Remove trailing dots

        # Remove -0, -1, -2, etc. suffixes
        if clean_name.endswith('-0') or clean_name.endswith('-1') or clean_name.endswith('-2'):
            clean_name = clean_name[:-2]

        # Only add to rename dict if the name actually changed
        if clean_name != name:
            rename_dict[name] = clean_name

    # Step 2: Apply renaming
    if rename_dict:
        try:
            raw.rename_channels(rename_dict)
            print(f"  Renamed {len(rename_dict)} channels: {list(rename_dict.keys())[:3]}...")
        except ValueError as e:
            print(f"  Warning: Channel renaming issue - {e}")
            # Continue anyway to see what channels we have

    # Step 3: Check for missing channels
    current_channels = raw.ch_names
    missing = [ch for ch in TARGET_CHANNELS if ch not in current_channels]

    if missing:
        print(f"  Missing channels: {missing}")
        print(f"  Available channels: {current_channels}")
        return None

    # Step 4: Select and reorder channels to match TARGET_CHANNELS exactly
    raw.pick_channels(TARGET_CHANNELS, ordered=True)

    return raw


def parse_chb_summary(summary_path: str, folder_name: str) -> pd.DataFrame:
    """
    By reading a single chbXX-summary.txt file
    translates Seizure information into pandas DataFrame.
    Supports both legacy and numbered seize formats.
    """

    records = []

    current_file = None
    seizure_count = 0
    start_time = None

    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()

            # File name
            if line.startswith("File Name:"):
                current_file = line.split(":")[1].strip()
                seizure_count = 0

            # Number of Seizures in File
            elif line.startswith("Number of Seizures in File:"):
                seizure_count = int(line.split(":")[1].strip())

            # Seizure start (general regex)
            elif seizure_count > 0 and re.search(r"Seizure\s+\d*\s*Start Time:", line):
                start_time = int(re.findall(r"\d+", line)[-1])

            # Seizure end (general regex)
            elif seizure_count > 0 and re.search(r"Seizure\s+\d*\s*End Time:", line):
                end_time = int(re.findall(r"\d+", line)[-1])

                records.append({
                    "folder": folder_name,
                    "file": current_file,
                    "seizure_start_sec": start_time,
                    "seizure_end_sec": end_time
                })

                start_time = None

    return pd.DataFrame(records)


def build_seizure_dataframe(dataset_path: str) -> pd.DataFrame:
    """
    By browsing through all chbXX folders
    seizure collects its information in a single DataFrame.
    """

    all_records = []

    dataset_path = Path(dataset_path)

    for folder in dataset_path.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name  # chb01, chb02, ...

        summary_file = folder / f"{folder_name}-summary.txt"

        if not summary_file.exists():
            continue

        df_folder = parse_chb_summary(
            summary_path=str(summary_file),
            folder_name=folder_name
        )

        if not df_folder.empty:
            all_records.append(df_folder)

    if not all_records:
        return pd.DataFrame(
            columns=["folder", "file", "seizure_start_sec", "seizure_end_sec"]
        )

    return pd.concat(all_records, ignore_index=True)


TARGET_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]


def standardize_channels(raw):
    """
    Renames, selects, and reorders channels to match TARGET_CHANNELS.
    """
    # Clean existing channel names
    # CHB-MIT files often have dots or suffixes (e.g., "T7-P7-0", "FZ-CZ.")
    rename_dict = {}
    for name in raw.ch_names:
        # Normalize: Uppercase -> Remove . -> Remove -0 or -1 artifacts
        clean_name = name.upper().replace('.', '').replace('-0', '').replace('-1', '')

        # Specific fix for some files where 'T8-P8' is labeled 'T8-P8-1'
        if clean_name != name:
            rename_dict[name] = clean_name

        # Apply renaming safely
        if rename_dict:
            try:
                raw.rename_channels(rename_dict)
            except ValueError as e:
                # Handle edge case where renaming creates duplicates
                pass

                # 2. Check availability
        current_channels = raw.ch_names
        missing = [ch for ch in TARGET_CHANNELS if ch not in current_channels]

        if missing:
            # You can choose to 'return None' to skip files, or continue with warning
            print(f" Missing channels: {missing}")
            return None

    raw.pick_channels(TARGET_CHANNELS, ordered=True)

    return raw


#########################################################################
##Preprocessing functions


def fix_eeg_channels_version_2(file_path, final_channels):
    try:
        # 1. ADIM: Veriyi yükleyerek oku (Unique isim sorunu için preload=True daha güvenli)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # MNE bazen "-" kanalını otomatik isimlendirir, onları temizleyelim
        # Sadece bizim istediğimiz final_channels listesindekileri tutalım
        current_channels = raw.ch_names

        # T8-P8 özel durum yönetimi (Eğer listede yoksa bile isimlendirme için bak)
        if 'T8-P8-1' in current_channels:
            raw.rename_channels({'T8-P8-1': 'T8-P8'})
        elif 'T8-P8-0' in current_channels and 'T8-P8' not in current_channels:
            raw.rename_channels({'T8-P8-0': 'T8-P8'})

        # Sadece final_channels içinde olanları seç, diğerlerini at
        available_to_keep = [ch for ch in final_channels if ch in raw.ch_names]
        raw.pick(available_to_keep)

        # Eksik kanal varsa doldur (Shape bozulmasın diye 0 ile doldurulmuş kanal ekler)
        missing_channels = set(final_channels) - set(raw.ch_names)
        if missing_channels:
            print(f"⚠️ {os.path.basename(file_path)} eksik kanallar: {missing_channels}. Sıfır verisi ekleniyor.")
            # Eksik kanalları 0 verisiyle ekleme yapısı (opsiyonel ama shape tutarlılığı için önemli)
            for m_ch in missing_channels:
                data = np.zeros((1, len(raw.times)))
                info = mne.create_info([m_ch], raw.info['sfreq'], ch_types='eeg')
                raw_extra = mne.io.RawArray(data, info)
                raw.add_channels([raw_extra])

        # Kanal sıralamasını sabitle (Modelin girişi için kritik!)
        raw.reorder_channels(final_channels)

        return raw
    except Exception as e:
        print(f"❌ Kanal hatası ({os.path.basename(file_path)}): {e}")
        return None


def fix_eeg_channels(file_path, final_channels):
    try:
        # 1. ADIM: Preload=False ile sadece meta veriyi oku (RAM kullanmaz)
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        ch_names = raw.ch_names

        # 2. ADIM: T8-P8 Özel Durumu ve Gereksiz Kanal Tespiti
        # Hem senin silmek istediğin kanalları hem de T8-P8 çakışmasını yönetiyoruz
        to_drop = [ch for ch in ch_names if ch not in final_channels and ch not in ['T8-P8-0', 'T8-P8-1']]

        if 'T8-P8-0' in ch_names and 'T8-P8-1' in ch_names:
            to_drop.append('T8-P8-0')

        # Gereksizleri atıyoruz (Hala RAM'e veri yüklemedik)
        raw.drop_channels(to_drop)

        # 3. ADIM: Veriyi Şimdi Yükle (Sadece kalan 23-24 kanalı yükler, RAM tasarrufu sağlar)
        raw.load_data(verbose=False)

        # 4. ADIM: İsimlendirme ve Sıralama
        if 'T8-P8-1' in raw.ch_names:
            raw.rename_channels({'T8-P8-1': 'T8-P8'})

        # Kanal Sıralamasını Sabitle
        raw.reorder_channels(final_channels)

        # Kontrol
        if len(raw.ch_names) != len(final_channels):
            print(f"⚠️ Uyarı: {os.path.basename(file_path)} için kanal sayısı eksik! "
                  f"Mevcut: {len(raw.ch_names)}")

        print(f"✅ {file_path} kanalları başarı ile çevirildi.")
        return raw

    except Exception as e:
        print(f"❌ Hata oluştu ({os.path.basename(file_path)}): {e}")
        return None


def apply_filtering_version2(raw):
    """
    Apply bandpass and notch filtering.
    - 5th-order Butterworth bandpass 0.5–60 Hz (zero-phase)
    - Notch filter at 60 Hz

    Parameters
    ----------
    raw : Raw
        MNE Raw object

    Returns
    -------
    Raw
        Filtered Raw object
    """
    # Bandpass filter: 0.5-60 Hz, 5th order Butterworth, zero-phase
    if not raw.preload:
        raw.load_data(verbose='error')

    raw_filtered = raw.copy().filter(
        l_freq=0.5,
        h_freq=60.0,
        method='iir',
        iir_params={'order': 5, 'ftype': 'butter'},
        phase='zero',
        verbose='error'
    )

    # Notch filter at 60 Hz
    raw_filtered.notch_filter(
        freqs=60.0,
        method='iir',
        verbose='error'
    )

    return raw_filtered

def apply_filtering(raw):
    """
    Apply bandpass and notch filtering.
    - 5th-order Butterworth bandpass 0.5–60 Hz (zero-phase)
    - Notch filter at 60 Hz

    Parameters
    ----------
    raw : Raw
        MNE Raw object

    Returns
    -------
    Raw
        Filtered Raw object
    """
    # Bandpass filter: 0.5-60 Hz, 5th order Butterworth, zero-phase
    raw_filtered = raw.copy().filter(
        l_freq=0.5,
        h_freq=60.0,
        method='iir',
        iir_params={'order': 5, 'ftype': 'butter'},
        phase='zero',
        verbose='error'
    )

    # Notch filter at 60 Hz
    raw_filtered.notch_filter(
        freqs=60.0,
        method='iir',
        verbose='error'
    )

    return raw_filtered


def downsample_version2(raw, TARGET_SFREQ=128.0):
    # Veri bellekte değilse resample hata verebilir veya çalışmayabilir
    if not raw.preload:
        raw.load_data()

    # Anti-aliasing filtresi otomatik uygulanır
    raw_resampled = raw.copy().resample(TARGET_SFREQ, npad='auto', verbose='error')
    return raw_resampled

def downsample(raw, TARGET_SFREQ=128.0):
    """
    Downsample from 256 Hz to 128 Hz.

    Parameters
    ----------
    raw : Raw
        MNE Raw object at 256 Hz

    Returns
    -------
    Raw
        Downsampled Raw object at 128 Hz
    """
    # Downsample using MNE's resample method
    raw_resampled = raw.copy().resample(TARGET_SFREQ, npad='auto')
    return raw_resampled


def save_single_processed_edf(raw_obj, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Dosya ismini belirle
    filename = os.path.basename(raw_obj.filenames[0]).replace('.edf', '_processed.edf')
    save_path = os.path.join(output_folder, filename)

    try:
        # Export işlemi
        raw_obj.export(save_path, fmt='edf', overwrite=True)
        print(f"💾 Başarıyla kaydedildi: {filename}")
    except Exception as e:
        print(f"❌ Kayıt hatası ({filename}): {e}")

def save_processed_edfs(processed_raw_list, output_folder="processed_files"):
    """
    processed_raw listesindeki objeleri belirtilen klasöre .edf olarak kaydeder.
    :param processed_raw_list: list | list of edf files has been processed !
    :output_folder : str | output folder path.
    usage:
    # save_processed_edfs(processed_raw)
    """
    # 1. Klasör yoksa oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Klasör oluşturuldu: {output_folder}")

    for raw_obj in processed_raw_list:
        # Dosya adını belirle (Eğer objenin içinde isim yoksa numara veriyoruz)
        filename = raw_obj.filenames[0].stem
        save_path = os.path.join(output_folder, filename)

        try:
            # mne kullanarak dışa aktar (export)
            # NOT: mne 1.2+ sürümlerinde .export() kullanılabilir.
            raw_obj.export(save_path, fmt='edf', overwrite=True)
            print(f"Başarıyla kaydedildi: {save_path}")
        except Exception as e:
            print(f"Hata oluştu ({filename}): {e}")
####################################################################


def build_seizure_annotations_for_file(df, file_name):
    """
    Belirli bir dosya adı için dataframe içindeki tüm seizure aralıklarını bulur
    ve bir MNE Annotations objesi oluşturur.
    """
    # Dosya adına göre filtreleme
    df_file = df[df["file"] == file_name]

    # Eğer dosyada hiç seizure yoksa boş bir annotation yapısı veya None döner
    if df_file.empty:
        return None, [], []

    onsets = []
    durations = []
    descriptions = []

    # Aynı dosya ismine sahip tüm seizure'lar üzerinde döner
    for _, row in df_file.iterrows():
        start = float(row["seizure_start_sec"])
        end = float(row["seizure_end_sec"])

        duration = end - start

        onsets.append(start)
        durations.append(duration)
        descriptions.append("seizure")

    # Tüm seizure'ları içeren tek bir Annotations objesi oluşturulur
    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions
    )

    return annotations, durations, descriptions

def generate_epoch_labels_version2(epochs, raw):
    """
    Epoch'ların seizure içerip içermediğini belirleyen label listesi üretir.
    NOT: raw objesi resample edilmişse, sfreq de güncel (128) olmalıdır.
    """
    sfreq = raw.info["sfreq"]  # Resample sonrası 128.0 olmalı
    ann = raw.annotations
    labels = []

    # Annotation aralıklarını hazırla
    seizure_intervals = []
    if ann is not None:
        for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
            if desc == "seizure":
                seizure_intervals.append((onset, onset + duration))

    # Her epoch için kontrol
    for event in epochs.events:
        # event[0] -> sample indisi. sfreq'e bölünce saniyeyi verir.
        epoch_start = event[0] / sfreq
        # epochs.tmax genelde 2.0 (duration) civarıdır.
        epoch_end = epoch_start + (epochs.tmax - epochs.tmin)

        label = 0
        for sz_start, sz_end in seizure_intervals:
            # Overlap (çakışma) kontrolü
            overlap = not (epoch_end <= sz_start or epoch_start >= sz_end)
            if overlap:
                label = 1
                break
        labels.append(label)

    return labels

def generate_epoch_labels(epochs, raw):
    """
    Epoch'ların seizure içerip içermediğini belirleyen label listesi üretir.
    """

    sfreq = raw.info["sfreq"]
    ann = raw.annotations

    labels = []

    # Annotation interval'ları hazırlanır
    seizure_intervals = []
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):

        if desc == "seizure":
            seizure_intervals.append((onset, onset + duration))

    # Her epoch için kontrol yapılır
    for event in epochs.events:

        epoch_start = event[0] / sfreq
        epoch_end = epoch_start + epochs.tmax - epochs.tmin

        label = 0

        # Overlap kontrolü
        for sz_start, sz_end in seizure_intervals:

            overlap = not (epoch_end <= sz_start or epoch_start >= sz_end)

            if overlap:
                label = 1
                break

        labels.append(label)

    return labels

##################
import os
import gc
import numpy as np
import mne
import utility.my_utils as deniz
import pandas as pd


def process_to_temp_files(list_preprocessed, df, temp_dir="temp_npy", epoch_length=10):
    """
    Her dosyayı tek tek işleyip ayrı .npy dosyaları olarak kaydeder.
    RAM dolmasını engeller ve veri tipini (float64) korur.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"📂 Geçici klasör oluşturuldu: {temp_dir}")

    processed_files = []

    for idx, edf_path in enumerate(list_preprocessed):
        # Dosya adını CSV ile eşleşecek şekilde düzenle
        file_name = os.path.basename(edf_path).replace('_processed.edf')

        try:
            # 1. Veriyi Yükle (Orijinal hassasiyet korunur)
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')

            # 2. Annotation Ekleme
            annotations, _, _ = deniz.build_seizure_annotations_for_file(df, file_name)
            if annotations is not None:
                raw.set_annotations(annotations)

            # 3. Epoch Oluşturma
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=epoch_length,
                preload=True,
                verbose='ERROR'
            )

            # 4. Label Üretimi
            labels = deniz.generate_epoch_labels(epochs, raw)

            # 5. Veriyi ve Label'ı al (float32 yapmadan, ham haliyle)
            X = epochs.get_data()  # Varsayılan float64
            y = np.array(labels)

            # 6. Diske Geçici Kayıt
            x_file = os.path.join(temp_dir, f"X_{idx}.npy")
            y_file = os.path.join(temp_dir, f"y_{idx}.npy")

            np.save(x_file, X)
            np.save(y_file, y)

            processed_files.append((x_file, y_file))
            print(f"✅ [{idx + 1}/{len(list_preprocessed)}] {file_name} -> {X.shape} kaydedildi.")

            # Temizlik
            del raw, epochs, X, y
            gc.collect()

        except Exception as e:
            print(f"❌ Hata: {file_name} -> {str(e)}")

    return processed_files

################ Verify processed
import numpy as np
import os
import random
import matplotlib.pyplot as plt


def verify_processed_data(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    if not files:
        print("❌ Klasörde .npz dosyası bulunamadı!")
        return

    # Rastgele bir dosya seç
    sample_file = random.choice(files)
    data = np.load(os.path.join(folder_path, sample_file))

    X = data['x']
    y = data['y']

    print(f"📄 Dosya: {sample_file}")
    print(f"📊 X Shape: {X.shape} (Epoch, Kanal, Sample)")
    print(f"🎯 y Shape: {y.shape} (Labels)")
    print(f"✅ Seizure Oranı: %{(sum(y) / len(y)) * 100:.2f} ({int(sum(y))} epoch)")

    # Eğer seizure varsa, ilk seizure epoch'unu görselleştir
    seizure_idxs = np.where(y == 1)[0]
    idx = seizure_idxs[0] if len(seizure_idxs) > 0 else 0

    plt.figure(figsize=(12, 6))
    for i in range(min(5, X.shape[1])):  # İlk 5 kanalı çizdir
        plt.plot(X[idx, i, :] + (i * 100))  # Kanalları üst üste binmesin diye kaydırdık

    label_text = "SEIZURE" if y[idx] == 1 else "HEALTHY"
    plt.title(f"{sample_file} - Epoch: {idx} - Label: {label_text}")
    plt.xlabel("Samples")
    plt.ylabel("Channels (Offset applied)")
    plt.show()