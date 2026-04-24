import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =============================================================================
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# =============================================================================
MODEL_PATH = r"C:\Users\gorse\Projelerim\Tez\best_model_subject_chb07.h5"
DATASET_PATH = "processed_master_datasets/master_dataset_10.0s.npz"

SUBJECT_ID = 7
EPOCH_SEC = 10.0
TARGET_WINDOW_SEC = 60.0
EPOCHS_PER_WINDOW = int(TARGET_WINDOW_SEC / EPOCH_SEC)
CHANNEL_TO_PLOT = 0
SAMPLING_RATE = 128

# ── AESTHETIC COLORS ──
BG_COLOR = "#0a0f1a"  # Deep dark blue/black background
GRID_COLOR = "#1e293b"  # Subtle grey-blue for grid
TEXT_COLOR = "#e2e8f0"  # Off-white for text
SIGNAL_COLOR = "#00e5ff"  # Neon Cyan for EEG
PROB_COLOR = "#ff2a5f"  # Neon Pink/Red for Probability
TRUE_LBL_COLOR = "#ffffff"  # White for true label highlights


# =============================================================================
# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
# =============================================================================
def find_consecutive_epochs(y_data, target_class, num_epochs):
    """Find starting index of `num_epochs` consecutive epochs of `target_class`."""
    for i in range(len(y_data) - num_epochs + 1):
        if np.all(y_data[i: i + num_epochs] == target_class):
            return i
    return None


def extract_and_predict(X_subj, y_subj, start_idx, model, num_epochs):
    """Extract consecutive epochs, concatenate the signal, and get predictions."""
    X_window = X_subj[start_idx: start_idx + num_epochs]
    y_window_true = y_subj[start_idx: start_idx + num_epochs]

    # Predict probabilities for each 10s epoch
    y_pred_probs = model.predict(X_window).flatten()

    # Concatenate the signals for plotting a continuous stream
    signal_concat = np.concatenate(X_window, axis=0)

    return signal_concat[:, CHANNEL_TO_PLOT], y_window_true, y_pred_probs


def plot_60s_window(ax, signal, true_labels, pred_probs, title):
    """Plot the continuous signal and overlay model predictions as a filled step graph."""
    time_axis = np.linspace(0, TARGET_WINDOW_SEC, len(signal))

    # --- Axis Styling ---
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.5)

    # --- 1. Plot Raw EEG Signal ---
    ax.plot(time_axis, signal, color=SIGNAL_COLOR, linewidth=1.2, alpha=0.9, label="EEG Signal")
    ax.set_title(title, fontsize=13, fontweight='bold', color=TEXT_COLOR, pad=15)
    ax.set_ylabel("Amplitude", color=TEXT_COLOR, fontsize=11, fontweight='semibold')
    ax.set_xlim([0, TARGET_WINDOW_SEC])
    ax.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, linewidth=1)

    # --- 2. Secondary Axis for Probability ---
    ax_pred = ax.twinx()
    ax_pred.set_ylim([-0.05, 1.05])
    ax_pred.set_ylabel("Seizure Probability", color=PROB_COLOR, fontsize=11, fontweight='bold')
    ax_pred.tick_params(colors=PROB_COLOR, labelsize=10)
    for spine in ax_pred.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.5)

    # Create step-coordinates for probabilities
    time_steps = np.arange(0, TARGET_WINDOW_SEC + EPOCH_SEC, EPOCH_SEC)

    # Append the last probability again to extend the final step to the edge of the graph
    probs_stepped = np.append(pred_probs, pred_probs[-1])

    # --- 3. Plot Probability Step & Fill ---
    ax_pred.fill_between(time_steps, 0, probs_stepped, step="post",
                         color=PROB_COLOR, alpha=0.20)
    ax_pred.step(time_steps, probs_stepped, where="post",
                 color=PROB_COLOR, linewidth=2.5, alpha=0.9, label="Predicted Probability")

    # --- 4. Subtle True Label Shading ---
    # Shades the background very faintly white where the ground truth is actually a seizure
    for i in range(EPOCHS_PER_WINDOW):
        if true_labels[i] == 1:
            ax.axvspan(time_steps[i], time_steps[i + 1], color=TRUE_LBL_COLOR, alpha=0.06,
                       label='Ground Truth (Seizure)' if i == 0 else "")

    return ax, ax_pred


# =============================================================================
# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
# =============================================================================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset not found at {DATASET_PATH}.")
        sys.exit(1)

    print(f"Loading data from {DATASET_PATH}...")
    data = np.load(DATASET_PATH)
    X = data['X']
    y = data['y']
    subjects = data['s']

    # Align X dimension with y
    N_samples = len(y)
    if X.shape[0] != N_samples:
        try:
            n_axis = list(X.shape).index(N_samples)
            X = np.moveaxis(X, n_axis, 0)
        except ValueError:
            print(f"❌ ERROR: Could not align X array dimensions.")
            sys.exit(1)

    # Filter for Subject
    subj_mask = (subjects == SUBJECT_ID) | (subjects == str(SUBJECT_ID)) | (subjects == f"chb{SUBJECT_ID:02d}")
    X_subj = X[subj_mask]
    y_subj = y[subj_mask]

    print(f"Subject {SUBJECT_ID:02d} data shape: {X_subj.shape}")

    # Find Windows
    idx_noseizure = find_consecutive_epochs(y_subj, target_class=0, num_epochs=EPOCHS_PER_WINDOW)
    idx_seizure = find_consecutive_epochs(y_subj, target_class=1, num_epochs=EPOCHS_PER_WINDOW)

    if idx_noseizure is None or idx_seizure is None:
        print("Warning: Could not find full 60s consecutive windows.")
        return

    print("Extracting segments and predicting...")
    sig_ns, true_ns, pred_ns = extract_and_predict(X_subj, y_subj, idx_noseizure, model, EPOCHS_PER_WINDOW)
    sig_sz, true_sz, pred_sz = extract_and_predict(X_subj, y_subj, idx_seizure, model, EPOCHS_PER_WINDOW)

    # --- FIGURE SETUP ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, facecolor=BG_COLOR)
    fig.subplots_adjust(hspace=0.35)

    # Plot Non-Seizure Window
    ax1, ax1_pred = plot_60s_window(axes[0], sig_ns, true_ns, pred_ns,
                                    title=f"SUBJECT: chb{SUBJECT_ID:02d} | 60s Window | Ground Truth: NO SEIZURE")

    # Plot Seizure Window
    ax2, ax2_pred = plot_60s_window(axes[1], sig_sz, true_sz, pred_sz,
                                    title=f"SUBJECT: chb{SUBJECT_ID:02d} | 60s Window | Ground Truth: SEIZURE")

    axes[1].set_xlabel("Time (Seconds)", color=TEXT_COLOR, fontsize=12, fontweight='bold')

    # Add a unified legend at the top
    lines, labels = ax1.get_legend_handles_labels()
    lines_pred, labels_pred = ax1_pred.get_legend_handles_labels()

    fig.legend(lines + lines_pred, labels + labels_pred,
               loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3,
               facecolor="#162032", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
               fontsize=11, framealpha=0.9)

    # Save the output plot
    save_path = "seizure_vs_noseizure_60s_cyberpunk.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ Plot successfully saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()