import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =============================================================================
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# =============================================================================
# --- 10-SECOND DATA & MODEL ---
MODEL_10S_PATH = r"best_model_subject_chb07_10s.h5"
DATASET_10S_PATH = "processed_master_datasets/master_dataset_10.0s.npz"

# --- 5-SECOND DATA & MODEL (UPDATE THESE PATHS) ---
MODEL_5S_PATH = r"best_model_subject_chb07_5s.h5"
DATASET_5S_PATH = "processed_master_datasets/master_dataset_5.0s.npz"

SUBJECT_ID = 7
TARGET_WINDOW_SEC = 60.0
CHANNEL_TO_PLOT = 0
DETECTION_THRESHOLD = 0.5

# ── AESTHETIC COLORS ──
BG_COLOR = "#0a0f1a"
GRID_COLOR = "#1e293b"
TEXT_COLOR = "#e2e8f0"
SIGNAL_COLOR = "#00e5ff"
PROB_COLOR = "#ff2a5f"
TRUE_LBL_COLOR = "#ffffff"


# =============================================================================
# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
# =============================================================================
def load_and_align_data(dataset_path):
    """Loads npz data and ensures the sample size (N) is on axis 0."""
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    data = np.load(dataset_path)
    X, y, subjects = data['X'], data['y'], data['s']

    N_samples = len(y)
    if X.shape[0] != N_samples:
        try:
            n_axis = list(X.shape).index(N_samples)
            X = np.moveaxis(X, n_axis, 0)
        except ValueError:
            print(f"❌ ERROR: Could not align X array dimensions for {dataset_path}.")
            sys.exit(1)

    # Filter for Subject
    subj_mask = (subjects == SUBJECT_ID) | (subjects == str(SUBJECT_ID)) | (subjects == f"chb{SUBJECT_ID:02d}")
    return X[subj_mask], y[subj_mask]


def extract_and_predict(X_subj, y_subj, start_idx, model, epochs_per_window):
    """Extract consecutive epochs, concatenate the signal, and get predictions."""
    X_window = X_subj[start_idx: start_idx + epochs_per_window]
    y_window_true = y_subj[start_idx: start_idx + epochs_per_window]

    y_pred_probs = model.predict(X_window, verbose=0).flatten()
    signal_concat = np.concatenate(X_window, axis=0)

    return signal_concat[:, CHANNEL_TO_PLOT], y_window_true, y_pred_probs


def plot_60s_window(ax, signal, true_labels, pred_probs, title, epoch_sec):
    """Plot the continuous signal and overlay model predictions as a filled step graph."""
    time_axis = np.linspace(0, TARGET_WINDOW_SEC, len(signal))
    epochs_per_window = int(TARGET_WINDOW_SEC / epoch_sec)

    # --- Axis Styling ---
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.5)

    # --- Plot Raw EEG Signal ---
    ax.plot(time_axis, signal, color=SIGNAL_COLOR, linewidth=1.2, alpha=0.9, label="EEG Signal")
    ax.set_title(title, fontsize=13, fontweight='bold', color=TEXT_COLOR, pad=15)
    ax.set_ylabel("Amplitude", color=TEXT_COLOR, fontsize=11, fontweight='semibold')
    ax.set_xlim([0, TARGET_WINDOW_SEC])

    # ── FIX: Dynamically set x-axis ticks based on the epoch size (5s or 10s) ──
    tick_locations = np.arange(0, TARGET_WINDOW_SEC + epoch_sec, epoch_sec)
    ax.set_xticks(tick_locations)

    ax.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, linewidth=1)

    # --- Secondary Axis for Probability ---
    ax_pred = ax.twinx()
    ax_pred.set_ylim([-0.05, 1.05])
    ax_pred.set_ylabel("Seizure Probability", color=PROB_COLOR, fontsize=11, fontweight='bold')
    ax_pred.tick_params(colors=PROB_COLOR, labelsize=10)
    for spine in ax_pred.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.5)

    # Plot Probability Step & Fill
    time_steps = np.arange(0, TARGET_WINDOW_SEC + epoch_sec, epoch_sec)
    probs_stepped = np.append(pred_probs, pred_probs[-1])

    ax_pred.fill_between(time_steps, 0, probs_stepped, step="post", color=PROB_COLOR, alpha=0.20)
    ax_pred.step(time_steps, probs_stepped, where="post", color=PROB_COLOR, linewidth=2.5, alpha=0.9,
                 label="Predicted Probability")

    # Threshold Line
    ax_pred.axhline(y=DETECTION_THRESHOLD, color='#ffdd57', linestyle=':', linewidth=1.5, alpha=0.8,
                    label="Detection Threshold")

    # True Label Shading
    for i in range(epochs_per_window):
        if true_labels[i] == 1:
            ax.axvspan(time_steps[i], time_steps[i + 1], color=TRUE_LBL_COLOR, alpha=0.06,
                       label='Ground Truth (Seizure)' if i == 0 else "")
    return ax, ax_pred


# =============================================================================
# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
# =============================================================================
def main():
    print("Loading Models...")
    if not os.path.exists(MODEL_10S_PATH) or not os.path.exists(MODEL_5S_PATH):
        print("❌ ERROR: One or both model paths are missing. Check Configuration.")
        sys.exit(1)

    model_10s = load_model(MODEL_10S_PATH)
    model_5s = load_model(MODEL_5S_PATH)

    print("Loading Datasets...")
    X_10s, y_10s = load_and_align_data(DATASET_10S_PATH)
    X_5s, y_5s = load_and_align_data(DATASET_5S_PATH)

    epochs_per_win_10s = int(TARGET_WINDOW_SEC / 10.0)
    epochs_per_win_5s = int(TARGET_WINDOW_SEC / 5.0)

    # ── Search for the Discrepancy Window ──
    print("Searching for a seizure window where 5s detects, but 10s misses...")
    found_idx_10s = None

    # Find all windows that are 100% true seizure
    for i in range(len(y_10s) - epochs_per_win_10s + 1):
        if np.all(y_10s[i: i + epochs_per_win_10s] == 1):

            # Map the 10s index to the 5s dataset (1 epoch of 10s = 2 epochs of 5s)
            idx_5s = i * 2

            # Get predictions for this specific window
            pred_10 = model_10s.predict(X_10s[i: i + epochs_per_win_10s], verbose=0).flatten()
            pred_5 = model_5s.predict(X_5s[idx_5s: idx_5s + epochs_per_win_5s], verbose=0).flatten()

            # Logic: Did 5s cross the threshold, but 10s failed to?
            if np.max(pred_5) >= DETECTION_THRESHOLD and np.max(pred_10) < DETECTION_THRESHOLD:
                found_idx_10s = i
                print(f"🎯 Discrepancy found at 10s index {i}!")
                break

    if found_idx_10s is None:
        print("⚠️ Warning: Could not find a window where 5s perfectly detects and 10s perfectly misses.")
        print("Defaulting to the very first seizure window found in the dataset.")
        # Fallback to first seizure
        for i in range(len(y_10s) - epochs_per_win_10s + 1):
            if np.all(y_10s[i: i + epochs_per_win_10s] == 1):
                found_idx_10s = i
                break

    if found_idx_10s is None:
        print("❌ ERROR: No 60-second seizure windows exist in this patient's data.")
        return

    # Extract data for plotting
    idx_5s = found_idx_10s * 2
    sig_10s, true_10s, pred_10s = extract_and_predict(X_10s, y_10s, found_idx_10s, model_10s, epochs_per_win_10s)
    sig_5s, true_5s, pred_5s = extract_and_predict(X_5s, y_5s, idx_5s, model_5s, epochs_per_win_5s)

    # --- FIGURE SETUP ---
    # ── FIX: Removed sharex=True so each plot can have its own customized timeline ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False, facecolor=BG_COLOR)
    fig.subplots_adjust(hspace=0.35)

    # Plot 5-Second Model
    ax1, ax1_pred = plot_60s_window(axes[0], sig_5s, true_5s, pred_5s, epoch_sec=5.0,
                                    title=f"SUBJECT: chb{SUBJECT_ID:02d} | 5-Second Epoch Model (Detection Successful)")

    # Plot 10-Second Model
    ax2, ax2_pred = plot_60s_window(axes[1], sig_10s, true_10s, pred_10s, epoch_sec=10.0,
                                    title=f"SUBJECT: chb{SUBJECT_ID:02d} | 10-Second Epoch Model (Detection Missed)")

    axes[1].set_xlabel("Time (Seconds)", color=TEXT_COLOR, fontsize=12, fontweight='bold')

    # Add a unified legend
    lines, labels = ax1.get_legend_handles_labels()
    lines_pred, labels_pred = ax1_pred.get_legend_handles_labels()

    # De-duplicate legend
    unique_labels = dict(zip(labels + labels_pred, lines + lines_pred))

    fig.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=4,
               facecolor="#162032", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
               fontsize=11, framealpha=0.9)

    # Save
    save_path = "seizure_5s_vs_10s_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ Plot successfully saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()