from scipy import signal
from scipy.signal import butter, filtfilt, welch, decimate
import numpy as np


class EEGPreprocessor:
    """
    Clean and prepare EEG signals for deep learning.
    """

    def __init__(self, sampling_rate=256, low_freq=0.5, high_freq=60, notch_freq=60, target_sfreq=128):
        self.fs = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        self.target_sfreq = target_sfreq

    def bandpass_filter(self, data, order=4):
        """Remove frequencies outside 0.5-60 Hz range."""
        nyquist = self.fs / 2
        low = self.low_freq / nyquist
        high = min(self.high_freq / nyquist, 0.99)

        b, a = butter(order, [low, high], btype='band')

        if len(data.shape) == 1:
            return filtfilt(b, a, data)
        else:
            return np.array([filtfilt(b, a, ch) for ch in data])


    def downsample(self, data):
        """
        Downsample data from self.fs to self.target_sfreq.
        """
        if self.fs == self.target_sfreq:
            return data

        # Downsampling oranı (Örn: 256 / 128 = 2)
        q = int(self.fs / self.target_sfreq)

        if len(data.shape) == 1:
            return decimate(data, q)
        else:
            # Her kanal için ayrı ayrı decimate uygular
            return np.array([decimate(ch, q) for ch in data])



    def notch_filter(self, data, Q=30):
        """Remove 60 Hz power line interference."""
        nyquist = self.target_sfreq / 2
        freq = self.notch_freq / nyquist

        b, a = signal.iirnotch(freq, Q)

        if len(data.shape) == 1:
            return filtfilt(b, a, data)
        else:
            return np.array([filtfilt(b, a, ch) for ch in data])

    def normalize(self, data):
        """Z-score normalization per channel."""
        if len(data.shape) == 1:
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        else:
            return np.array([(ch - np.mean(ch)) / (np.std(ch) + 1e-8) for ch in data])

    def preprocess(self, data):
        """Apply full preprocessing pipeline."""
        # Step 1: Bandpass filter
        data = self.bandpass_filter(data)
        # Step 2: Downsample filter
        data = self.downsample(data)
        # Step 3: Notch filter
        data = self.notch_filter(data)
        # Step 4: Normalize
        data = self.normalize(data)

        return data
