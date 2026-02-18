import numpy as np
import pandas as pd
import mne
import os


## Fix channels
def fix_eeg_channels(file_path, final_channels, verbose=False):
    try:

        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        ch_names = raw.ch_names

        #, T8-P8 Özel Durumu ve Gereksiz Kanal Tespiti
        to_drop = [ch for ch in ch_names if ch not in final_channels and ch not in ['T8-P8-0', 'T8-P8-1']]

        if 'T8-P8-0' in ch_names and 'T8-P8-1' in ch_names:
            to_drop.append('T8-P8-0')

        # drop
        raw.drop_channels(to_drop)

        # İsimlendirme ve Sıralama
        if 'T8-P8-1' in raw.ch_names:
            raw.rename_channels({'T8-P8-1': 'T8-P8'})

        # Kanal Sıralamasını Sabitle
        raw.reorder_channels(final_channels)

        # Kontrol
        if len(raw.ch_names) != len(final_channels):
            print(f"⚠️ Uyarı: {os.path.basename(file_path)} için kanal sayısı eksik! "
                  f"Mevcut: {len(raw.ch_names)}")

        if verbose:
            print(f"✅ {file_path} kanalları başarı ile çevirildi.")

        return raw

    except Exception as e:
        print(f"❌ Hata oluştu ({os.path.basename(file_path)}): {e}")
        return None


#filtering
def apply_filtering_version(raw):
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
        print('this file was not preloaded')

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


# downsample
def downsample_version(raw, TARGET_SFREQ=128.0):
    # Veri bellekte değilse resample hata verebilir veya çalışmayabilir
    if not raw.preload:
        raw.load_data()
        print('this file was not preloaded')

    # Anti-aliasing filtresi otomatik uygulanır
    raw_resampled = raw.copy().resample(TARGET_SFREQ, npad='auto', verbose='error')
    return raw_resampled