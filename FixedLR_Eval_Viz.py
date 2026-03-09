import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

# =============================================================================
# ── CONFIGURATION  ────────────────────────────────────────────────────────────
# =============================================================================
TIMESTAMP   = "20260304-135510-1s"
OUTPUT_DIR  = os.path.join("saved_outputs", TIMESTAMP)
suffix = TIMESTAMP.split('-')[-1]

# Path to CSV produced by the evaluation script
SUMMARY_CSV = os.path.join(OUTPUT_DIR, f"overall_loso_summary-{suffix}.csv")

# Where to save the output figure
OUTPUT_FIG  = os.path.join(OUTPUT_DIR, "plots", f"loso_results_table_{suffix}.png")

DECISION_THRESHOLD = 0.3

# Columns to display in the table (must match CSV column names)
METRICS = [
    ("auroc",             "AUROC"),
    ("auprc",             "AUPRC\n(PRC)"),
    ("sensitivity",       "Sensitivity\n(Se)"),
    ("specificity",       "Specificity\n(Sp)"),
    ("balanced_accuracy", "Balanced\nAccuracy"),
    ("seizure_f1",        "Seizure\nF1"),
]

# =============================================================================
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "metric_cmap",
    [(0.00, "#d62728"),   # red   – poor
     (0.50, "#ff7f0e"),   # orange
     (0.65, "#ffdd57"),   # yellow
     (0.80, "#74c476"),   # light green
     (1.00, "#238b45")],  # dark green – excellent
)

def score_color(value: float, alpha: float = 0.25) -> tuple:
    """Return an RGBA face colour for a metric cell."""
    rgba = list(CMAP(float(np.clip(value, 0, 1))))
    rgba[3] = alpha
    return tuple(rgba)

def text_color(value: float) -> str:
    """Dark text for bright cells, light text for dark cells."""
    return "#1a1a2e" if value >= 0.60 else "#ffffff"


# =============================================================================
# ── DATA LOADING  ─────────────────────────────────────────────────────────────
# =============================================================================
def load_summary(output_dir: str) -> pd.DataFrame:
    """
    Load the LOSO summary CSV directly from the experiment's output folder.
    Uses glob to find the file dynamically, bypassing naming inconsistencies (e.g. -4s vs _5s)
    and entirely removes the synthetic data fallback.
    """
    # Look for any file starting with 'overall_loso_summary' and ending in '.csv'
    search_pattern = os.path.join(output_dir, "overall_loso_summary*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        raise FileNotFoundError(
            f" No summary CSV found in '{output_dir}'. Please ensure the evaluation script has run.")

    # Take the first matching CSV found in the folder
    actual_csv_path = csv_files[0]
    print(f" Loading dataset from: {actual_csv_path}")

    df = pd.read_csv(actual_csv_path)
    df["subject"] = df["subject"].astype(str)

    return df

# =============================================================================
# ── STATS FOOTER  ─────────────────────────────────────────────────────────────
# =============================================================================
def build_footer(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 4-row DataFrame with mean / std / min / max per metric."""
    cols = [m[0] for m in METRICS]
    rows = []
    for stat, fn in [("μ  Mean", "mean"), ("σ  Std", "std"),
                     ("↓  Min",  "min"),  ("↑  Max", "max")]:
        row = {"subject": stat}
        for c in cols:
            row[c] = getattr(df[c], fn)()
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# ── MAIN PLOT  ────────────────────────────────────────────────────────────────
# =============================================================================
def plot_results_table(df: pd.DataFrame, save_path: str) -> None:
    """
    Render a colour-coded per-subject metrics table and save as PNG.

    Layout
    ------
    • Header row      – metric names
    • Data rows       – one per subject, cells colour-coded by score
    • Mini bar        – horizontal bar inside each cell
    • Footer rows     – mean / std / min / max with bold mean row
    • Legend strip    – colour scale explanation
    • Stats panel     – aggregate mean ± std for each metric (below table)
    """
    footer_df  = build_footer(df)
    n_subjects = len(df)
    n_metrics  = len(METRICS)

    # ── Figure geometry ───────────────────────────────────────────────────────
    row_h      = 0.52          # inches per data row
    header_h   = 0.80          # inches for the column-header row
    footer_h   = 0.48 * 4      # 4 footer rows
    legend_h   = 0.55
    stats_h    = 1.20
    col_w      = [1.8] + [1.55] * n_metrics   # subject col + metric cols

    fig_w  = sum(col_w) + 0.6
    fig_h  = header_h + n_subjects * row_h + footer_h + legend_h + stats_h + 0.9

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0a0f1a")

    # ── Axes for the table body ───────────────────────────────────────────────
    margin_l = 0.25 / fig_w
    margin_r = 1 - 0.25 / fig_w
    top_rel  = 1 - 0.45 / fig_h
    table_h_rel = (header_h + n_subjects * row_h + footer_h) / fig_h

    ax = fig.add_axes([margin_l, top_rel - table_h_rel, margin_r - margin_l, table_h_rel])
    ax.set_facecolor("#0a0f1a")
    ax.axis("off")

    total_rows = 1 + n_subjects + 4        # header + data + footer
    total_cols = 1 + n_metrics

    # Normalised column widths
    col_fracs = np.array(col_w) / sum(col_w)
    col_x     = np.concatenate([[0], np.cumsum(col_fracs)])  # left edges

    row_heights = (
        [header_h] +
        [row_h]    * n_subjects +
        [0.48]     * 4
    )
    total_h_pt = sum(row_heights)
    row_y       = np.concatenate([[0], np.cumsum(row_heights)])  # bottom edges (pt)
    # Convert to axes fraction (top-to-bottom)
    row_y_frac  = (total_h_pt - row_y) / total_h_pt             # flip: 0=top

    def cell_rect(row_i, col_i):
        """Return (x, y, w, h) in axes coords for a cell."""
        x = col_x[col_i]
        w = col_fracs[col_i]
        y = row_y_frac[row_i + 1]           # bottom of cell
        h = row_heights[row_i] / total_h_pt
        return x, y, w, h

    # ── Helper: draw one cell ─────────────────────────────────────────────────
    def draw_cell(row_i, col_i, text, facecolor, textcolor,
                  fontsize=9.5, fontweight="normal", fontfamily="monospace",
                  value=None, is_header=False):
        x, y, w, h = cell_rect(row_i, col_i)
        pad = 0.004

        # Background rectangle
        rect = FancyBboxPatch(
            (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
            boxstyle="round,pad=0.003",
            facecolor=facecolor,
            edgecolor="#1e293b",
            linewidth=0.5,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)

        # Mini bar (only for metric data cells, not header/subject/footer)
        if value is not None and col_i > 0 and not is_header:
            bar_h   = 0.18 * h
            bar_y   = y + pad + 0.06 * h
            bar_x0  = x + pad + 0.04 * w
            bar_w   = (w - 2 * pad) * 0.88
            # Background track
            bg = FancyBboxPatch(
                (bar_x0, bar_y), bar_w, bar_h,
                boxstyle="round,pad=0.001",
                facecolor="#1e293b", edgecolor="none",
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(bg)
            # Filled portion
            fill_w = bar_w * float(np.clip(value, 0, 1))
            if fill_w > 0:
                fill = FancyBboxPatch(
                    (bar_x0, bar_y), fill_w, bar_h,
                    boxstyle="round,pad=0.001",
                    facecolor=CMAP(float(np.clip(value, 0, 1))),
                    edgecolor="none",
                    transform=ax.transAxes, clip_on=False,
                )
                ax.add_patch(fill)

        # Text
        ax.text(
            x + w / 2,
            y + h / 2 + (0.13 * h if value is not None and col_i > 0 and not is_header else 0),
            text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=textcolor, fontfamily=fontfamily,
            transform=ax.transAxes,
        )

    # ── Row 0: Header ─────────────────────────────────────────────────────────
    draw_cell(0, 0, "Subject", "#162032", "#94a3b8",
              fontsize=9, fontweight="bold", is_header=True)
    for j, (_, label) in enumerate(METRICS):
        draw_cell(0, j + 1, label, "#162032", "#94a3b8",
                  fontsize=8.5, fontweight="bold", is_header=True)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for i, (_, row) in enumerate(df.iterrows()):
        ri = i + 1
        bg = "#0d1526" if i % 2 == 0 else "#0a0f1a"
        draw_cell(ri, 0, str(row["subject"]), bg, "#38bdf8",
                  fontsize=9, fontfamily="monospace")
        for j, (col, _) in enumerate(METRICS):
            val  = float(row[col])
            fc   = score_color(val, alpha=0.30)
            tc   = CMAP(float(np.clip(val, 0, 1)))
            draw_cell(ri, j + 1, f"{val:.4f}", fc, tc,
                      fontsize=9, fontfamily="monospace",
                      fontweight="semibold", value=val)

    # ── Footer rows ───────────────────────────────────────────────────────────
    footer_styles = [
        ("#16213e", "#818cf8", "bold",   11.0),   # mean
        ("#0f172a", "#475569", "normal",  9.0),   # std
        ("#0f172a", "#475569", "normal",  9.0),   # min
        ("#0f172a", "#475569", "normal",  9.0),   # max
    ]
    for fi, (_, frow) in enumerate(footer_df.iterrows()):
        ri = 1 + n_subjects + fi
        fc_bg, tc_subj, fw, fs = footer_styles[fi]
        draw_cell(ri, 0, str(frow["subject"]), fc_bg, tc_subj,
                  fontsize=fs, fontweight=fw, fontfamily="monospace")
        for j, (col, _) in enumerate(METRICS):
            val = float(frow[col])
            if fi == 0:   # mean row – colour-code
                fc = score_color(val, alpha=0.25)
                tc = CMAP(float(np.clip(val, 0, 1)))
            else:
                fc, tc = fc_bg, "#475569"
            draw_cell(ri, j + 1, f"{val:.4f}", fc, tc,
                      fontsize=fs, fontweight=fw, fontfamily="monospace")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 1 - 0.12 / fig_h,
        f"LOSO Evaluation — CHB-MIT Seizure Detection · {suffix}-seconds Epoch",
        ha="center", va="top",
        fontsize=14, fontweight="bold", color="#f8fafc",
        fontfamily="monospace",
    )

    # ── Legend strip ──────────────────────────────────────────────────────────
    legend_bottom = (top_rel - table_h_rel) - legend_h / fig_h - 0.05 / fig_h
    ax_leg = fig.add_axes([margin_l, legend_bottom,
                           margin_r - margin_l, legend_h / fig_h])
    ax_leg.set_facecolor("#0a0f1a")
    ax_leg.axis("off")

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_leg.imshow(gradient, aspect="auto", cmap=CMAP,
                  extent=[0, 1, 0, 1], transform=ax_leg.transAxes,
                  alpha=0.85)
    for xv, lbl in [(0.0, "0.00"), (0.25, "0.25"), (0.50, "0.50"),
                    (0.75, "0.75"), (1.0, "1.00")]:
        ax_leg.text(xv, -0.55, lbl, ha="center", va="top",
                    fontsize=8, color="#94a3b8", transform=ax_leg.transAxes)
    ax_leg.text(0.0, 1.0, "Poor", ha="left", va="bottom",
                fontsize=7.5, color="#d62728", transform=ax_leg.transAxes)
    ax_leg.text(1.0, 1.0, "Excellent", ha="right", va="bottom",
                fontsize=7.5, color="#238b45", transform=ax_leg.transAxes)

    # ── Aggregate stats panel ─────────────────────────────────────────────────
    stats_bottom = legend_bottom - stats_h / fig_h - 0.08 / fig_h
    ax_st = fig.add_axes([margin_l, stats_bottom,
                          margin_r - margin_l, stats_h / fig_h])
    ax_st.set_facecolor("#0f172a")
    ax_st.axis("off")
    ax_st.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02",
        facecolor="#0f172a", edgecolor="#1e293b", linewidth=1,
        transform=ax_st.transAxes, clip_on=False,
    ))

    ax_st.text(0.015, 0.92, "Aggregate Statistics  (mean ± std  |  min – max)",
               ha="left", va="top", fontsize=9, fontweight="bold",
               color="#818cf8", transform=ax_st.transAxes,
               fontfamily="monospace")

    col_positions = np.linspace(0.015, 0.98, n_metrics + 1)[:-1]
    for j, (col, label) in enumerate(METRICS):
        xp  = col_positions[j]
        lbl = label.replace("\n", " ")
        mn  = df[col].mean()
        sd  = df[col].std()
        mi  = df[col].min()
        mx  = df[col].max()
        ax_st.text(xp, 0.73, lbl, ha="left", va="top",
                   fontsize=8, color="#94a3b8", transform=ax_st.transAxes)
        ax_st.text(xp, 0.50,
                   f"{mn:.4f} ± {sd:.4f}",
                   ha="left", va="top", fontsize=9.5, fontweight="bold",
                   color=CMAP(float(np.clip(mn, 0, 1))),
                   transform=ax_st.transAxes, fontfamily="monospace")
        ax_st.text(xp, 0.22,
                   f"[{mi:.4f} – {mx:.4f}]",
                   ha="left", va="top", fontsize=8, color="#475569",
                   transform=ax_st.transAxes, fontfamily="monospace")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅  Figure saved → {save_path}")


# =============================================================================
# ── OPTIONAL: print formatted table to console  ───────────────────────────────
# =============================================================================
def print_console_table(df: pd.DataFrame) -> None:
    footer_df = build_footer(df)
    cols      = [m[0] for m in METRICS]
    labels    = [m[1].replace("\n", " ") for m in METRICS]

    col_w_subj = 10
    col_w_met  = 14
    header_row = f"{'Subject':<{col_w_subj}}" + "".join(f"{l:>{col_w_met}}" for l in labels)

    sep = "─" * len(header_row)
    print(f"\n{'═' * len(header_row)}")
    print("LOSO PER-SUBJECT RESULTS")
    print(f"{'═' * len(header_row)}")
    print(header_row)
    print(sep)

    for _, row in df.iterrows():
        line = f"{str(row['subject']):<{col_w_subj}}"
        for c in cols:
            line += f"{row[c]:>{col_w_met}.4f}"
        print(line)

    print(sep)
    for _, frow in footer_df.iterrows():
        line = f"{str(frow['subject']):<{col_w_subj}}"
        for c in cols:
            line += f"{frow[c]:>{col_w_met}.4f}"
        print(line)
    print(f"{'═' * len(header_row)}\n")

# =============================================================================
# ── ENTRY POINT  ─────────────────────────────────────────────────────────────
# =============================================================================
if __name__ == "__main__":
    # Pass the folder directory instead of the exact CSV path
    summary_df = load_summary(OUTPUT_DIR)

    print_console_table(summary_df)

    plot_results_table(summary_df, OUTPUT_FIG)

    # Optional: also export a clean formatted CSV
    export_cols = ["subject"] + [m[0] for m in METRICS]
    export_path = os.path.join(OUTPUT_DIR, f"loso_results_table_{suffix}.csv")
    summary_df[export_cols].round(4).to_csv(export_path, index=False)
    print(f"✅  Formatted CSV saved → {export_path}")