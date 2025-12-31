"""
Multi-scale sliding window extraction for EEG data.

This module provides functions to extract sliding windows of different lengths
from continuous EEG data, aligned to a common time grid.
"""

import numpy as np
from typing import Dict, List


def create_multi_scale_windows(
    data: np.ndarray,
    time_indices: np.ndarray,
    window_lengths_sec: List[float],
    sfreq: float
) -> Dict[str, np.ndarray]:
    """
    Extract multi-scale sliding windows from continuous EEG data.
    
    Each window of length L seconds ends at the corresponding time index.
    For example, if time_index corresponds to t=10s and window_length=2s,
    the window spans from t=8s to t=10s.
    
    Parameters
    ----------
    data : np.ndarray
        Raw EEG data of shape (n_channels, n_samples)
    time_indices : np.ndarray
        Array of time indices (in samples) where each window ends.
        Shape: (n_timepoints,)
    window_lengths_sec : List[float]
        List of window lengths in seconds (e.g., [1.0, 2.0, 5.0, 8.0, 10.0])
    sfreq : float
        Sampling frequency in Hz
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping window length labels (e.g., '2s') to window arrays.
        Each array has shape (n_timepoints, n_channels, n_samples_in_window)
        
    Example
    -------
    >>> data = np.random.randn(22, 100000)  # 22 channels, ~13 minutes at 128 Hz
    >>> time_indices = np.arange(128, 100000, 128)  # Every 1 second
    >>> windows_dict = create_multi_scale_windows(
    ...     data=data,
    ...     time_indices=time_indices,
    ...     window_lengths_sec=[1.0, 2.0, 5.0],
    ...     sfreq=128.0
    ... )
    >>> print(windows_dict['2s'].shape)  # (n_timepoints, 22, 256)
    """
    n_channels, n_total_samples = data.shape
    n_timepoints = len(time_indices)
    
    windows_dict = {}
    
    for window_length_sec in window_lengths_sec:
        # Calculate window length in samples
        window_length_samples = int(window_length_sec * sfreq)
        
        # Create label (e.g., '2s', '5s')
        label = f"{int(window_length_sec)}s" if window_length_sec.is_integer() else f"{window_length_sec}s"
        
        # Pre-allocate array for all windows
        windows = np.zeros((n_timepoints, n_channels, window_length_samples), dtype=np.float32)
        
        # Extract windows
        for i, end_idx in enumerate(time_indices):
            start_idx = end_idx - window_length_samples
            
            # Boundary check: skip if window extends before data start
            if start_idx < 0:
                # Pad with zeros if needed (or you could skip this window entirely)
                padding = -start_idx
                windows[i, :, padding:] = data[:, 0:end_idx]
                # First `padding` samples remain zero
            else:
                # Normal case: extract window
                windows[i, :, :] = data[:, start_idx:end_idx]
        
        windows_dict[label] = windows
    
    return windows_dict


def validate_windows(windows_dict: Dict[str, np.ndarray], sfreq: float, expected_channels: int = 22):
    """
    Validate that extracted windows have correct shapes and properties.
    
    Parameters
    ----------
    windows_dict : Dict[str, np.ndarray]
        Dictionary of window arrays from create_multi_scale_windows
    sfreq : float
        Sampling frequency in Hz
    expected_channels : int
        Expected number of channels (default: 22 for CHB-MIT)
        
    Raises
    ------
    ValueError
        If validation fails
    """
    for label, windows in windows_dict.items():
        # Extract window length from label (e.g., '2s' -> 2.0)
        window_length_sec = float(label.rstrip('s'))
        expected_samples = int(window_length_sec * sfreq)
        
        # Check shape
        n_windows, n_channels, n_samples = windows.shape
        
        if n_channels != expected_channels:
            raise ValueError(f"Window '{label}': Expected {expected_channels} channels, got {n_channels}")
        
        if n_samples != expected_samples:
            raise ValueError(
                f"Window '{label}': Expected {expected_samples} samples "
                f"({window_length_sec}s at {sfreq} Hz), got {n_samples}"
            )
        
        # Check for NaN or Inf
        if np.any(np.isnan(windows)) or np.any(np.isinf(windows)):
            raise ValueError(f"Window '{label}' contains NaN or Inf values")
    
    print(f"✅ Window validation passed for {len(windows_dict)} scales")
