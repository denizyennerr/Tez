"""
CHB-MIT EEG Preprocessing Module (Deterministic, Benchmark-Comparable)

Conservative, reproducible preprocessing pipeline for CHB-MIT:
- Load EDF files from base_dir
- Objective bad-channel detection (variance + peak-to-peak)
- Drop bad channels (no interpolation)
- Align channels to fixed canonical bipolar order (supports duplicates)
- Filtering: 5th-order Butterworth bandpass 0.5–60 Hz (zero-phase)
- Notch filtering: 60 Hz
- Downsampling: 256 Hz -> 128 Hz
- Parse CHB-MIT summary files for seizure annotations
- Data selection:
    * 10-patient subset (default matches common modern benchmark subsets)
    * interictal defined as >= 4 hours away from any seizure
- Multi-scale sliding window extraction for decision-fusion:
    at each step t, extract last {2,4,8,10} seconds ending at t
- Save outputs as .npz with metadata in data/preprocessed

Notes:
- This module intentionally does NOT do ICA/ASR/manual cleaning.
- Bad channel handling is objective and deterministic.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import mne
from mne.io import Raw


class CHBMITPreprocessor:
    """
    Deterministic preprocessing class for CHB-MIT Scalp EEG dataset.
    
    Provides conservative, benchmark-comparable preprocessing with:
    - Objective bad channel detection and removal
    - Fixed channel order alignment
    - Standard filtering pipeline
    - Multi-scale sliding window extraction
    - Interictal data selection (>= 4 hours from seizures)
    """
    
    # Canonical channel order for CHB-MIT (standard bipolar montage)
    CANONICAL_CHANNELS = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ',
        'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'
    ]
    
    # Default 10-patient subset 
    DEFAULT_SUBSET = [
        'chb01'
    ]
    
    # Multi-scale window lengths (seconds)
    WINDOW_LENGTHS = [2.0, 5.0, 8.0, 10.0]
    
    # Target sampling rate after downsampling
    TARGET_SFREQ = 128.0  # Hz
    
    # Source sampling rate (CHB-MIT standard)
    SOURCE_SFREQ = 256.0  # Hz
    
    # Interictal threshold (4 hours in seconds)
    INTERICTAL_THRESHOLD_SEC = 4 * 3600  # 14400 seconds
    
    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        subject_subset: Optional[List[str]] = None,
        bad_channel_variance_threshold: float = 1e-10,
        bad_channel_flat_threshold: float = 1e-6,
        interictal_threshold_sec: float = 4 * 3600
    ):
        """
        Initialize CHB-MIT preprocessor.
        
        Parameters
        ----------
        base_dir : str
            Path to CHB-MIT dataset root directory (contains chb01/, chb02/, etc.)
        output_dir : str
            Path to output directory for preprocessed data
        subject_subset : Optional[List[str]]
            List of subject IDs to process. If None, uses DEFAULT_SUBSET (10 patients)
        bad_channel_variance_threshold : float
            Minimum variance threshold for bad channel detection (default: 1e-10)
        bad_channel_flat_threshold : float
            Maximum peak-to-peak threshold for flat channel detection (default: 1e-6)
        interictal_threshold_sec : float
            Minimum time away from seizures to be considered interictal (default: 4 hours)
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.subject_subset = self.DEFAULT_SUBSET
        self.bad_channel_variance_threshold = bad_channel_variance_threshold
        self.bad_channel_flat_threshold = bad_channel_flat_threshold
        self.interictal_threshold_sec = interictal_threshold_sec
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for parsed summary files
        self._summary_cache: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        
        # Cache for all seizure times per subject (for interictal selection)
        self._all_seizure_times_cache: Dict[str, List[Tuple[float, float]]] = {}
        
        # Cache for global seizure times (absolute time across all files)
        self._global_seizure_times_cache: Dict[str, List[Tuple[float, float]]] = {}
        
        # Cache for file durations and cumulative offsets
        self._file_durations_cache: Dict[str, Dict[str, float]] = {}
        self._cumulative_offsets_cache: Dict[str, Dict[str, float]] = {}
    
    def parse_summary_file(self, subject_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse CHB-MIT summary file to extract seizure annotations.
        
        Parameters
        ----------
        subject_dir : Path
            Path to subject directory (e.g., chb01/)
            
        Returns
        -------
        Dict[str, List[Tuple[float, float]]]
            Dictionary mapping EDF filenames to list of (start, end) seizure times in seconds
        """
        subject_id = subject_dir.name
        
        # Check cache
        if subject_id in self._summary_cache:
            return self._summary_cache[subject_id]
        
        summary_file = subject_dir / f"{subject_id}-summary.txt"
        
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")
        
        seizure_map: Dict[str, List[Tuple[float, float]]] = {}
        current_file: Optional[str] = None
        
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect filename
            if line.startswith("File Name:"):
                current_file = line.split("File Name:")[1].strip()
                if current_file not in seizure_map:
                    seizure_map[current_file] = []
            
            # Detect seizure start time
            if current_file and "Seizure Start Time" in line:
                try:
                    # Extract start time in seconds
                    start_match = re.search(r':\s*(\d+)\s*seconds', line)
                    if start_match:
                        start_sec = float(start_match.group(1))
                        
                        # Look ahead for end time
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            end_match = re.search(r':\s*(\d+)\s*seconds', next_line)
                            if end_match:
                                end_sec = float(end_match.group(1))
                                seizure_map[current_file].append((start_sec, end_sec))
                except (AttributeError, ValueError, IndexError):
                    # Skip malformed entries
                    continue
        
        # Cache result
        self._summary_cache[subject_id] = seizure_map
        
        return seizure_map
    
    def get_file_durations(self, subject_id: str) -> Dict[str, float]:
        """
        Get file durations by loading EDF files.
        Used for calculating cumulative time offsets.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'chb01')
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping EDF filenames to durations in seconds
        """
        if subject_id in self._file_durations_cache:
            return self._file_durations_cache[subject_id]
        
        subject_dir = self.base_dir / subject_id
        edf_files = sorted(subject_dir.glob("*.edf"))
        
        durations = {}
        for edf_path in edf_files:
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose='error')
                durations[edf_path.name] = raw.times[-1]
            except Exception:
                # If file can't be loaded, estimate from summary or skip
                durations[edf_path.name] = 0.0
        
        self._file_durations_cache[subject_id] = durations
        return durations
    
    def get_cumulative_offsets(self, subject_id: str) -> Dict[str, float]:
        """
        Calculate cumulative time offsets for each file.
        First file starts at 0, subsequent files start at cumulative sum of previous durations.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'chb01')
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping EDF filenames to cumulative start offsets in seconds
        """
        if subject_id in self._cumulative_offsets_cache:
            return self._cumulative_offsets_cache[subject_id]
        
        durations = self.get_file_durations(subject_id)
        subject_dir = self.base_dir / subject_id
        edf_files = sorted(subject_dir.glob("*.edf"))
        
        offsets = {}
        cumulative = 0.0
        
        for edf_path in edf_files:
            edf_filename = edf_path.name
            offsets[edf_filename] = cumulative
            cumulative += durations.get(edf_filename, 0.0)
        
        self._cumulative_offsets_cache[subject_id] = offsets
        return offsets
    
    def get_global_seizure_times(self, subject_id: str) -> List[Tuple[float, float]]:
        """
        Get all seizure times in absolute (global) time across all files.
        Seizure times are converted from file-relative to absolute time.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'chb01')
            
        Returns
        -------
        List[Tuple[float, float]]
            List of (start, end) seizure times in absolute seconds
        """
        if subject_id in self._global_seizure_times_cache:
            return self._global_seizure_times_cache[subject_id]
        
        seizure_map = self.parse_summary_file(self.base_dir / subject_id)
        offsets = self.get_cumulative_offsets(subject_id)
        
        global_seizures = []
        for edf_filename, file_seizures in seizure_map.items():
            file_offset = offsets.get(edf_filename, 0.0)
            for start_sec, end_sec in file_seizures:
                global_seizures.append((file_offset + start_sec, file_offset + end_sec))
        
        # Sort by start time
        global_seizures.sort(key=lambda x: x[0])
        
        self._global_seizure_times_cache[subject_id] = global_seizures
        return global_seizures
    
    def get_all_seizure_times(self, subject_id: str) -> List[Tuple[float, float]]:
        """
        Get all seizure times across all files for a subject.
        Used for interictal selection.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'chb01')
            
        Returns
        -------
        List[Tuple[float, float]]
            List of (start, end) seizure times in seconds (relative to file start)
        """
        if subject_id in self._all_seizure_times_cache:
            return self._all_seizure_times_cache[subject_id]
        
        subject_dir = self.base_dir / subject_id
        seizure_map = self.parse_summary_file(subject_dir)
        
        # Collect all seizure times
        all_seizures = []
        for file_seizures in seizure_map.values():
            all_seizures.extend(file_seizures)
        
        self._all_seizure_times_cache[subject_id] = all_seizures
        return all_seizures
    
    def is_interictal(self, time_sec: float, seizure_times: List[Tuple[float, float]]) -> bool:
        """
        Check if a time point is interictal (>= threshold away from any seizure).
        
        Parameters
        ----------
        time_sec : float
            Time point in seconds
        seizure_times : List[Tuple[float, float]]
            List of (start, end) seizure times
            
        Returns
        -------
        bool
            True if interictal, False otherwise
        """
        for start_sec, end_sec in seizure_times:
            # Check if time is within seizure
            if start_sec <= time_sec <= end_sec:
                return False
            
            # Check if time is too close to seizure (within threshold)
            distance_before = time_sec - end_sec if time_sec > end_sec else float('inf')
            distance_after = start_sec - time_sec if time_sec < start_sec else float('inf')
            
            min_distance = min(distance_before, distance_after)
            if min_distance < self.interictal_threshold_sec:
                return False
        
        return True
    
    def detect_bad_channels(self, raw: Raw) -> List[str]:
        """
        Objective bad channel detection based on variance and flatness.
        
        Parameters
        ----------
        raw : Raw
            MNE Raw object
            
        Returns
        -------
        List[str]
            List of bad channel names
        """
        bad_channels = []
        data = raw.get_data()

        
        for idx, ch_name in enumerate(raw.ch_names):
            ch_data = data[idx, :]
            
            # Check variance (too low indicates dead/flat channel)
            variance = np.var(ch_data)
            if variance < self.bad_channel_variance_threshold:
                bad_channels.append(ch_name)
                continue
            
            # Check peak-to-peak (too low indicates flat channel)
            ptp = np.ptp(ch_data)
            if ptp < self.bad_channel_flat_threshold:
                bad_channels.append(ch_name)
                continue
        
        return bad_channels
    
    
    def apply_filtering(self, raw: Raw) -> Raw:
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
    
    def downsample(self, raw: Raw) -> Raw:
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
        if abs(raw.info['sfreq'] - self.SOURCE_SFREQ) > 1.0:
            raise ValueError(
                f"Expected source sampling rate {self.SOURCE_SFREQ} Hz, "
                f"got {raw.info['sfreq']} Hz"
            )
        
        # Downsample using MNE's resample method
        raw_resampled = raw.copy().resample(self.TARGET_SFREQ, npad='auto')
        
        return raw_resampled
    
    def extract_multi_scale_windows(
        self,
        raw: Raw,
        seizure_times: List[Tuple[float, float]],
        global_seizure_times: List[Tuple[float, float]],
        cumulative_start_time: float
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Extract multi-scale sliding windows for decision fusion.
        At each time step t, extract last {2,5,8,10} seconds ending at t.
        
        Parameters
        ----------
        raw : Raw
            MNE Raw object
        seizure_times : List[Tuple[float, float]]
            Seizure times for this file (relative to file start)
        global_seizure_times : List[Tuple[float, float]]
            All seizure times in absolute (global) time across all files
        cumulative_start_time : float
            Cumulative start time of this file (absolute time)
            
        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray]
            (windows_dict, labels) where:
            - windows_dict: {window_length: array} with shape (n_timepoints, n_channels, n_samples)
            - labels: binary array of shape (n_timepoints,) where:
                1 = seizure, 0 = interictal, -1 = preictal/postictal (exclude)
        """
        data = raw.get_data()  # Shape: (n_channels, n_samples)
        sfreq = raw.info['sfreq']
        n_channels, n_total_samples = data.shape
        
        # Calculate time points (every second, ending at each second)
        # Extract windows ending at each second
        time_points = np.arange(1.0, raw.times[-1] + 1.0, 1.0)  # Every 1 second
        n_timepoints = len(time_points)
        
        # Initialize windows dictionary
        windows_dict = {}
        for window_length in self.WINDOW_LENGTHS:
            window_samples = int(window_length * sfreq)
            windows_dict[f'{int(window_length)}s'] = np.zeros(
                (n_timepoints, n_channels, window_samples),
                dtype=np.float32
            )
        
        # Initialize labels
        labels = np.zeros(n_timepoints, dtype=np.int32)
        
        # Extract windows and labels
        for t_idx, t_end in enumerate(time_points):
            # Convert to absolute time
            t_abs = cumulative_start_time + t_end
            
            # Check if this time point is in a seizure (using file-relative times for speed)
            is_seizure = False
            for start_sec, end_sec in seizure_times:
                if start_sec <= t_end <= end_sec:
                    is_seizure = True
                    labels[t_idx] = 1
                    break
            
            # If not seizure, check if interictal using global seizure times
            if not is_seizure:
                if self.is_interictal(t_abs, global_seizure_times):
                    labels[t_idx] = 0  # Interictal
                else:
                    labels[t_idx] = -1  # Preictal/postictal (exclude from training)
            
            # Extract windows ending at t_end
            for window_length in self.WINDOW_LENGTHS:
                window_samples = int(window_length * sfreq)
                t_start = t_end - window_length
                
                # Calculate sample indices
                start_sample = int(t_start * sfreq)
                end_sample = int(t_end * sfreq)
                
                # Handle boundary conditions
                if start_sample < 0:
                    # Pad with zeros at the beginning
                    pad_samples = -start_sample
                    end_sample = min(end_sample, n_total_samples)
                    window_data = data[:, 0:end_sample]
                    padding = np.zeros((n_channels, pad_samples))
                    window_data = np.concatenate([padding, window_data], axis=1)
                elif end_sample > n_total_samples:
                    # Pad with zeros at the end
                    pad_samples = end_sample - n_total_samples
                    start_sample = max(0, start_sample)
                    window_data = data[:, start_sample:n_total_samples]
                    padding = np.zeros((n_channels, pad_samples))
                    window_data = np.concatenate([window_data, padding], axis=1)
                else:
                    window_data = data[:, start_sample:end_sample]
                
                # Ensure correct length
                if window_data.shape[1] < window_samples:
                    padding = np.zeros((n_channels, window_samples - window_data.shape[1]))
                    window_data = np.concatenate([window_data, padding], axis=1)
                elif window_data.shape[1] > window_samples:
                    window_data = window_data[:, :window_samples]
                
                windows_dict[f'{int(window_length)}s'][t_idx] = window_data
        
        return windows_dict, labels
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization with outlier clipping.
        Clips outliers first to prevent artifacts from skewing the mean.
        
        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            Normalized data array
        """
       
        # Clip outliers first to prevent artifacts from skewing the mean
        data_clipped = np.clip(data, -500, 500)
        
        # Z-score normalization per channel
        mean = np.mean(data_clipped, axis=1, keepdims=True)
        std = np.std(data_clipped, axis=1, keepdims=True)
        
        #end region
        data_normalized = (data_clipped - mean) / (std + 1e-6)  # Add epsilon to avoid div by zero
        

        return data_normalized
    
    def preprocess_file(
        self,
        edf_path: Path,
        seizure_times: List[Tuple[float, float]],
        global_seizure_times: List[Tuple[float, float]],
        cumulative_start_time: float
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Preprocess a single EDF file.
        
        Parameters
        ----------
        edf_path : Path
            Path to EDF file
        seizure_times : List[Tuple[float, float]]
            List of (start, end) seizure times in seconds (relative to file start)
        global_seizure_times : List[Tuple[float, float]]
            All seizure times in absolute (global) time across all files
        cumulative_start_time : float
            Cumulative start time of this file (absolute time)
            
        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray]
            (windows_dict, labels) where windows_dict contains multi-scale windows
        """
        # 1. Load EDF
        raw = mne.io.read_raw_edf(
            str(edf_path),
            preload=True,
            verbose='error'
        )
        
        # 2. Detect and drop bad channels (no interpolation)
        bad_channels = self.detect_bad_channels(raw)
        if bad_channels:
            raw.drop_channels(bad_channels)
        
        # 3. Apply filtering (bandpass + notch)
        raw = self.apply_filtering(raw)
        
        # 4. Downsample (256 Hz → 128 Hz)
        raw = self.downsample(raw)
        
        # 5. Z-score normalization (per-file, per-channel)
        data = raw.get_data()
        data_normalized = self.normalize_data(data)
        
        # Update raw object with normalized data
        raw._data = data_normalized
        
        # 6. Extract multi-scale sliding windows
        windows_dict, labels = self.extract_multi_scale_windows(
            raw, seizure_times, global_seizure_times, cumulative_start_time
        )
        
        return windows_dict, labels
    
    def preprocess_subject(self, subject_id: str) -> Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """
        Preprocess all EDF files for a subject.
        Tracks cumulative time across files for global seizure tracking.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'chb01')
            
        Returns
        -------
        Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]]
            Dictionary mapping EDF filenames to (windows_dict, labels) tuples
        """
        subject_dir = self.base_dir / subject_id
        
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        
        # Parse summary file for seizure annotations
        seizure_map = self.parse_summary_file(subject_dir)
        
        # Get global seizure times (absolute time across all files)
        global_seizure_times = self.get_global_seizure_times(subject_id)
        
        # Get cumulative offsets for each file
        cumulative_offsets = self.get_cumulative_offsets(subject_id)
        
        # Get all EDF files
        edf_files = sorted(subject_dir.glob("*.edf"))
        
        results = {}
        
        for edf_path in edf_files:
            edf_filename = edf_path.name
            
            # Get seizure times for this file (empty list if no seizures)
            seizure_times = seizure_map.get(edf_filename, [])
            
            # Get cumulative start time for this file
            cumulative_start_time = cumulative_offsets.get(edf_filename, 0.0)
            
            try:
                windows_dict, labels = self.preprocess_file(
                    edf_path, seizure_times, global_seizure_times, cumulative_start_time
                )
                results[edf_filename] = (windows_dict, labels)
                
            except Exception as e:
                print(f"Error processing {edf_filename}: {e}")
                continue
        
        return results

    
    def canonical_channels(self, raw: Raw) -> Raw:
        """
        Select and reorder channels to match the canonical list.
        Renames slightly different channels (e.g., 'T8-P8-1' -> 'T8-P8') 
        to ensure consistency across patients.
        """
        # 1. Standardize current channel names (remove spaces, dots, -1 suffixes)
        # This handles the messy naming in CHB-MIT (e.g. "FP1-F7." vs "FP1-F7")
        # rename_map = {}
        rename_map = {ch: ch.replace('.', '').strip() for ch in raw.ch_names}
        # Handle duplicates like 'T8-P8-1' -> 'T8-P8'
        rename_map = {ch: ch.replace('-0', '').replace('-1', '') for ch in rename_map if ch.replace('-0', '').replace('-1', '') in self.CANONICAL_CHANNELS}
        
        # Apply renaming
        if rename_map:
            raw.rename_channels(rename_map)

        # 2. Check for missing channels
        current_channels = raw.ch_names
        missing = [ch for ch in self.CANONICAL_CHANNELS if ch not in current_channels]
        
        if len(missing) > 0:
            # For a thesis, strictness is good. Raise error if critical channels are missing.
            raise ValueError(f"Subject is missing canonical channels: {missing}")
            
        # 3. Pick and Reorder
        # This ensures the matrix shape is always (23, Time)
        raw_ordered = raw.copy().pick_channels(self.CANONICAL_CHANNELS, ordered=True)
        
        return raw_ordered
      
    
    def save_preprocessed(
        self,
        subject_id: str,
        results: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]]
    ):
        """
        Save preprocessed data to output directory.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'chb01')
        results : Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]]
            Dictionary mapping EDF filenames to (windows_dict, labels) tuples
        """
        subject_output_dir = self.output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        for edf_filename, (windows_dict, labels) in results.items():
            # Create output filename (remove .edf extension, add .npz)
            output_filename = edf_filename.replace('.edf', '.npz')
            output_path = subject_output_dir / output_filename
            
            # Prepare metadata
            metadata = {
                'sfreq': self.TARGET_SFREQ,
                'n_channels': len(self.CANONICAL_CHANNELS),
                'channel_names': self.CANONICAL_CHANNELS,
                'window_lengths': self.WINDOW_LENGTHS,
                'interictal_threshold_sec': self.interictal_threshold_sec,
                'labels': labels
            }
            
            # Add all window scales to save dict
            save_dict = {**windows_dict, **metadata}
            
            # Save as compressed numpy array
            np.savez_compressed(str(output_path), **save_dict)



# ============================================================================
# Data Splitting Functions (for binary classification)
# ============================================================================

def create_patient_split(
    subject_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test split by patient (Leave-One-Subject-Out style).
    Ensures no data leakage by splitting at patient boundaries.
    
    Parameters
    ----------
    subject_ids : List[str]
        List of subject IDs to split (e.g., ['chb01', 'chb02', ...])
    train_ratio : float
        Ratio of patients for training (default: 0.8)
    val_ratio : float
        Ratio of patients for validation (default: 0.1)
    test_ratio : float
        Ratio of patients for testing (default: 0.1)
    random_seed : Optional[int]
        Random seed for reproducibility (default: None)
        
    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        (train_subjects, val_subjects, test_subjects)
        
    Examples
    --------
    >>> subjects = ['chb01', 'chb02', ..., 'chb10']
    >>> train, val, test = create_patient_split(subjects)
    >>> # Train on patients 1-8, Val on patient 9, Test on patient 10
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Shuffle subject IDs for random split
    shuffled_subjects = subject_ids.copy()
    if random_seed is not None:
        np.random.shuffle(shuffled_subjects)
    else:
        # Deterministic shuffle based on subject IDs
        shuffled_subjects = sorted(shuffled_subjects)
    
    n_subjects = len(shuffled_subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    n_test = n_subjects - n_train - n_val  # Remaining goes to test
    
    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train:n_train + n_val]
    test_subjects = shuffled_subjects[n_train + n_val:]
    
    return train_subjects, val_subjects, test_subjects


def load_preprocessed_windows(
    preprocessed_dir: Path,
    subject_ids: List[str],
    window_scale: str = '5s',
    filter_labels: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed windows from .npz files for specified subjects.
    
    Parameters
    ----------
    preprocessed_dir : Path
        Path to preprocessed data directory
    subject_ids : List[str]
        List of subject IDs to load
    window_scale : str
        Window scale to load (e.g., '2s', '5s', '8s', '10s')
    filter_labels : bool
        If True, only return windows with labels 0 (interictal) or 1 (seizure).
        Excludes preictal/postictal windows (label -1).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (windows, labels) where:
        - windows: shape (n_windows, n_channels, n_samples)
        - labels: shape (n_windows,) with 0=interictal, 1=seizure
    """
    all_windows = []
    all_labels = []
    
    for subject_id in subject_ids:
        subject_dir = preprocessed_dir / subject_id
        
        if not subject_dir.exists():
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue
        
        # Load all .npz files for this subject
        npz_files = sorted(subject_dir.glob("*.npz"))
        
        for npz_path in npz_files:
            try:
                data = np.load(npz_path)
                
                # Extract windows for specified scale
                if window_scale not in data:
                    print(f"Warning: Window scale {window_scale} not found in {npz_path}")
                    continue
                
                windows = data[window_scale]  # Shape: (n_timepoints, n_channels, n_samples)
                labels = data['labels']  # Shape: (n_timepoints,)
                
                # Filter labels if requested
                if filter_labels:
                    mask = (labels == 0) | (labels == 1)
                    windows = windows[mask]
                    labels = labels[mask]
                    # Convert to binary: 0=interictal, 1=seizure
                    labels = (labels == 1).astype(np.int32)
                
                all_windows.append(windows)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue
    
    if len(all_windows) == 0:
        raise ValueError("No windows loaded from specified subjects")
    
    # Concatenate all windows
    windows_array = np.concatenate(all_windows, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    return windows_array, labels_array

