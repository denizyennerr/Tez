import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import mne
from mne.io import Raw
import gc
from eeg.windowing import create_multi_scale_windows


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
        'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
    
    # Default 10-patient subset 
    DEFAULT_SUBSET = [
        'chb01', 
        'chb03',
        'chb05',
        'chb09',
        'chb10',
        'chb14',
        'chb19',
        'chb20',
        'chb21',
        'chb23',
    ]
    
    # Multi-scale window lengths (seconds)
    WINDOW_LENGTHS = [1.0, 2.0, 5.0, 8.0, 10.0]
    
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
        bad_channel_high_amp_threshold: float = 500e-6,  
        interictal_threshold_sec: float = 4 * 3600,
        use_stratified_sampling: bool = False,
        seizure_stride_sec: float = 1.0,
        interictal_stride_sec: float = 10.0
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
        self.subject_subset = subject_subset if subject_subset else self.DEFAULT_SUBSET
        
        # Artifact Thresholds
        self.bad_channel_variance_threshold = bad_channel_variance_threshold
        self.bad_channel_flat_threshold = bad_channel_flat_threshold
        self.bad_channel_high_amp_threshold = bad_channel_high_amp_threshold
        
        self.interictal_threshold_sec = interictal_threshold_sec
        self.use_stratified_sampling = use_stratified_sampling
        self.seizure_stride_sec = seizure_stride_sec
        self.interictal_stride_sec = interictal_stride_sec

        # Cache initialization
        self._summary_cache: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self._all_seizure_times_cache: Dict[str, List[Tuple[float, float]]] = {}
        self._global_seizure_times_cache: Dict[str, List[Tuple[float, float]]] = {}
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
            Subject ID (e.g., 'chb01', 'chb03', 'chb05', 'chb09', 'chb10', 'chb14', 'chb19', 'chb20', 'chb21', 'chb23')
            
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

    def canonical_channels(self, raw: Raw) -> Raw:
        """
        Enforces canonical channels by pre-calculating all indices 
        and performing a SINGLE pick operation at the end.
        """
        print(f"DEBUG: Initial channels ({len(raw.ch_names)}): {raw.ch_names}")

        # 1. Map Clean Names to Original Indices
        available_channels = {}
        for idx, ch in enumerate(raw.ch_names):
            # Clean: remove -0, -1 suffixes, dots, spaces
            clean = re.sub(r'-\d+$', '', ch).replace('.', '').strip()
            
            if clean not in available_channels:
                available_channels[clean] = []
            available_channels[clean].append((ch, idx))

        indices_to_keep = []
        rename_map = {}

        # 2. Build the list of indices to keep (Pull method)
        for target_name in self.CANONICAL_CHANNELS:
            if target_name in available_channels:
                candidates = available_channels[target_name]
                
                # Rule: Always take the FIRST occurrence (e.g., T8-P8-0 at idx 14)
                # This implicitly drops the second occurrence (T8-P8-1 at idx 22)
                chosen_name, chosen_idx = candidates[0]
                
                indices_to_keep.append(chosen_idx)
                rename_map[chosen_name] = target_name
                
                # Debug log for duplicates
                if len(candidates) > 1:
                    print(f"  ! Duplicate found for {target_name}. Keeping idx {chosen_idx}, Dropping others.")
            else:
                raise ValueError(f"Missing required canonical channel: {target_name}")

        # --- CRITICAL: The operations below must be OUTSIDE the loop ---
        
        print(f"DEBUG: Picking {len(indices_to_keep)} indices: {indices_to_keep}")
        
        # 3. Perform the Atomic Pick (Removes T8-P8-1 and other unused channels)
        raw.pick(indices_to_keep)
        
        # 4. Rename (Renames T8-P8-0 back to T8-P8)
        raw.rename_channels(rename_map)
        
        # 5. Reorder
        raw.reorder_channels(self.CANONICAL_CHANNELS)

        print(f"SUCCESS: Final channels ({len(raw.ch_names)}): {raw.ch_names}")
        
        return raw

    def detect_bad_channels(self, raw: Raw) -> List[str]:
        """
        Only drops channels that are technically broken (flat/disconnected).
        Does NOT drop channels just because they are noisy/muscular.
        """
        bad_channels = []
        data = raw.get_data()
        
        for idx, ch_name in enumerate(raw.ch_names):
            ch_data = data[idx, :]
            
            # Only drop if the signal is essentially a flat line (Dead sensor)
            if np.var(ch_data) < 1e-15:  # Extremely low variance
                bad_channels.append(ch_name)

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
        # Downsample using MNE's resample method
        raw_resampled = raw.copy().resample(self.TARGET_SFREQ, npad='auto') 
        return raw_resampled

    def normalize_window(self, window_data: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization with outlier clipping.
        Clips outliers first to prevent artifacts from skewing the mean.
        
        Parameters
        ----------
        window_data : np.ndarray
            Data array of shape (n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            Normalized data array
        """
       
        # Clip outliers first to prevent artifacts from skewing the mean
        data_clipped = np.clip(window_data, -500, 500)
        
        # Z-score normalization per channel
        mean = np.mean(data_clipped, axis=1, keepdims=True)
        std = np.std(data_clipped, axis=1, keepdims=True)
        
        #end region
        data_normalized = (data_clipped - mean) / (std + 1e-18)  # Add epsilon to avoid div by zero
        
        return data_normalized

    def extract_multi_scale_windows(self, raw, seizure_times, global_seizure_times, cumulative_start_time):
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # 1. Define the Time Grid (Every 1 second)
        time_points_sec = np.arange(1.0, raw.times[-1] + 1.0, 1.0)
        time_indices = (time_points_sec * sfreq).astype(int)
        n_timepoints = len(time_points_sec)

        # 2. Generate Labels
        labels = np.zeros(n_timepoints, dtype=np.int32)
        
        for i, t_end in enumerate(time_points_sec):
            t_abs = cumulative_start_time + t_end
            
            # Check Seizure
            is_seizure = False
            for start, end in seizure_times:
                if start <= t_end <= end:
                    labels[i] = 1
                    is_seizure = True
                    break
            
            # Check Interictal
            if not is_seizure:
                if self.is_interictal(t_abs, global_seizure_times):
                    labels[i] = 0
                else:
                    labels[i] = -1 # Exclude
                    
        # 3. Generate Raw Windows (Delegated to windowing.py)
        print(f"   -> Slicing windows for {n_timepoints} timepoints...")
        windows_dict = create_multi_scale_windows(
            data=data,
            time_indices=time_indices,
            window_lengths_sec=self.WINDOW_LENGTHS,
            sfreq=sfreq
        )

        # 4. Apply Normalization (Crucial Step Added)
        # We apply Z-score normalization to the whole batch efficiently
        print(f"   -> Normalizing windows...")

        for key, windows_batch in windows_dict.items():            
            # Clip outliers (vectorized)
            windows_batch = np.clip(windows_batch, -500, 500)
            
            # Calculate mean/std along the time axis (axis=2)
            mean = np.mean(windows_batch, axis=2, keepdims=True)
            std = np.std(windows_batch, axis=2, keepdims=True)
            
            # Z-Score
            windows_dict[key] = (windows_batch - mean) / (std + 1e-18)

        return windows_dict, labels
            
    def preprocess_file(
        self,
        edf_path: Path,
        seizure_times: List[Tuple[float, float]],
        global_seizure_times: List[Tuple[float, float]],
        cumulative_start_time: float
    ):
        """
        Preprocess a single EDF file: load, clean, filter, downsample, and window.
        
        Parameters
        ----------
        edf_path : Path
            Path to the EDF file
        seizure_times : List[Tuple[float, float]]
            Seizure times for this file (file-relative seconds)
        global_seizure_times : List[Tuple[float, float]]
            All seizure times for the subject (absolute seconds)
        cumulative_start_time : float
            Cumulative time offset for this file (seconds)
            
        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray] or (None, None)
            windows_dict: Dictionary mapping window lengths to arrays
            labels: Label array for each window
        """
        try:
            # 1. Load EDF file
            print(f"     -> Loading EDF...")
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose='error')
            
            # 2. Align to canonical channels
            print(f"     -> Aligning to canonical channels...")
            raw = self.canonical_channels(raw)
            
            # 3. Detect and remove bad channels
            print(f"     -> Detecting bad channels...")
            bad_channels = self.detect_bad_channels(raw)
            if bad_channels:
                print(f"        ! Removing bad channels: {bad_channels}")
                raw.drop_channels(bad_channels)
                # If critical channels were removed, we might skip this file
                if len(raw.ch_names) < 18:  # Keep files with at least 18 good channels
                    print(f"        ⚠️ Too many bad channels ({len(bad_channels)}), skipping file")
                    return None, None
            
            # 4. Apply filtering
            print(f"     -> Applying bandpass (0.5-60 Hz) and notch (60 Hz) filters...")
            raw = self.apply_filtering(raw)
            
            # 5. Downsample to 128 Hz
            print(f"     -> Downsampling to {self.TARGET_SFREQ} Hz...")
            raw = self.downsample(raw)
            
            # 6. Extract multi-scale windows
            print(f"     -> Extracting multi-scale windows...")
            windows_dict, labels = self.extract_multi_scale_windows(
                raw=raw,
                seizure_times=seizure_times,
                global_seizure_times=global_seizure_times,
                cumulative_start_time=cumulative_start_time
            )
            
            # 7. Clean up
            del raw
            gc.collect()
            
            return windows_dict, labels
            
        except Exception as e:
            print(f"        ❌ Error in preprocess_file: {e}")
            return None, None

    def save_preprocessed(self, subject_id, edf_filename, windows_dict, labels):
            subject_output_dir = self.output_dir / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = subject_output_dir / edf_filename.replace('.edf', '.npz')
            
            metadata = {
                'sfreq': self.TARGET_SFREQ,
                'channel_names': self.CANONICAL_CHANNELS,
                'labels': labels
            }
            np.savez_compressed(str(output_path), **{**windows_dict, **metadata})
            print(f"  -> Saved {output_path.name}")

    def save_subject(self, subject_id: str):
        """
        Process all EDF files for a single subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'chb01')
        """
        subject_dir = self.base_dir / subject_id

        # Check if subject directory exists
        if not subject_dir.exists():
            print(f"  ❌ Error: Subject directory not found: {subject_dir}")
            return
        
        print(f"\nProcessing Subject: {subject_id}")
        
        # Load subject metadata once
        seizure_map = self.parse_summary_file(subject_dir)
        global_seizure_times = self.get_global_seizure_times(subject_id)
        cumulative_offsets = self.get_cumulative_offsets(subject_id)
        
        # Process each EDF file
        for edf_path in sorted(subject_dir.glob("*.edf")):
            print(f"  > Processing {edf_path.name}...")
            try:
                windows_dict, labels = self.preprocess_file(
                    edf_path, 
                    seizure_map.get(edf_path.name, []), 
                    global_seizure_times, 
                    cumulative_offsets.get(edf_path.name, 0.0)
                )
                
                if windows_dict is None:
                    continue

                self.save_preprocessed(subject_id, edf_path.name, windows_dict, labels)
                
                del windows_dict, labels
                gc.collect()

            except Exception as e:
                print(f"  ❌ Error processing {edf_path.name}: {e}")

# ============================================================================
# Data Splitting Functions
# ============================================================================

def create_patient_split(
    subject_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split subjects into train, validation, and test sets.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    shuffled_subjects = list(subject_ids).copy() # Ensure it's a list
    if random_seed is not None:
        np.random.shuffle(shuffled_subjects)
    else:
        shuffled_subjects = sorted(shuffled_subjects)
    
    n_subjects = len(shuffled_subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
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
    Load and concatenate preprocessed windows for a list of subjects.
    """
    all_windows = []
    all_labels = []
    
    # Ensure preprocessed_dir is a Path object
    preprocessed_dir = Path(preprocessed_dir)
    
    for subject_id in subject_ids:
        subject_dir = preprocessed_dir / subject_id
        
        if not subject_dir.exists():
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue
        
        npz_files = sorted(subject_dir.glob("*.npz"))
        
        for npz_path in npz_files:
            try:
                data = np.load(npz_path)
                
                if window_scale not in data:
                    print(f"Warning: Window scale {window_scale} not found in {npz_path}")
                    continue
                
                windows = data[window_scale]
                
                # NOTE: This assumes your preprocessing saved a 'labels' key.
                # If your current preprocessing doesn't save labels, this line will fail later.
                if 'labels' in data:
                    labels = data['labels']
                    
                    if filter_labels:
                        mask = (labels == 0) | (labels == 1)
                        windows = windows[mask]
                        labels = labels[mask]
                        labels = (labels == 1).astype(np.int32)
                    
                    all_labels.append(labels)
                else:
                    # Fallback if no labels exist (e.g. unsupervised training)
                    if filter_labels: 
                        print(f"Warning: No 'labels' found in {npz_path}, skipping label filtering.")
                
                all_windows.append(windows)
                
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue
    
    if len(all_windows) == 0:
        # Return empty arrays instead of crashing if specific subjects have no data
        return np.array([]), np.array([])
    
    windows_array = np.concatenate(all_windows, axis=0)
    
    if all_labels:
        labels_array = np.concatenate(all_labels, axis=0)
    else:
        labels_array = np.array([])
    
    return windows_array, labels_array