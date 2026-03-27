import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ⚙️ Configuration & Academic Styling
# =============================================================================
RESULTS_CSV = "saved_outputs_play/ensemble_results_final/decision_fusion_macro_mean.csv"
OUTPUT_DIR = os.path.join("saved_outputs_play", "ensemble_results_final", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Academic styling: Serif fonts are standard for thesis documents
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
})

# Set a clean, white background without heavy gridlines
sns.set_theme(style="ticks", context="paper")

# =============================================================================
# 1. Load Data
# =============================================================================
if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Could not find {RESULTS_CSV}. Please ensure it is in the correct directory.")

df = pd.read_csv(RESULTS_CSV)
df = df.drop(columns=['Decision Threshold', 'N_Subjects'], axis=1)

# Melt DataFrame for Seaborn plotting (Grouped Bar Chart format)
df_melted = df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

# =============================================================================
# 2. Grouped Bar Chart (All Models Comparison)
# =============================================================================
plt.figure(figsize=(16, 7))

# Use a colorblind-friendly palette; add edge colors for crisp bar boundaries
ax = sns.barplot(
    data=df_melted,
    x="Metric",
    y="Score",
    hue="Model",
    palette="colorblind",
    edgecolor="black",
    linewidth=0.7,
    capsize=0.05
)

# Note: In a thesis, titles are often omitted in favor of LaTeX/Word figure captions.
# You can comment out the next line if you prefer caption-only descriptions.
plt.title("Performance Comparison Across Epoched & Ensemble Models", pad=15)
plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
plt.ylim(0.0, 1.05)
plt.xticks(rotation=15)

# Add a subtle grid on the y-axis to help read values
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
sns.despine(trim=True, offset=5) # Removes top/right borders for a cleaner look

# Place the legend outside the plot cleanly
plt.legend(title="Model Type", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, edgecolor='black')
plt.tight_layout()

# Save plot (using bbox_inches='tight' prevents legend cutoff)
barplot_path = os.path.join(OUTPUT_DIR, "all_models_comparison_barplot.png")
plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
print(f"✅ Comparison Barplot saved to: {barplot_path}")
plt.close()

# =============================================================================
# 3. Overall Performance Heatmap
# =============================================================================
plt.figure(figsize=(12, 8))

# Set the Model as the index for the heatmap
df_heatmap = df.set_index("Model")

# Create Heatmap
# Switched to YlGnBu (Yellow-Green-Blue) which is standard in academia,
# prints well in grayscale, and avoids red/green colorblindness issues.
ax_heat = sns.heatmap(
    df_heatmap,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    cbar_kws={'label': 'Score'},
    vmin=0.2, vmax=1.0,
    linewidths=0.5,       # Adds borders between cells
    linecolor='lightgray' # Keeps the borders subtle
)

# Note: Similarly, consider removing the title if using a document caption.
plt.title("Overall Performance Metrics Across All Models", pad=15)
plt.ylabel("Model")
plt.xlabel("Evaluation Metric")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0) # Ensure y-labels are strictly horizontal

plt.tight_layout()

# Save plot
heatmap_path = os.path.join(OUTPUT_DIR, "all_models_comparison_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✅ Comparison Heatmap saved to: {heatmap_path}")
plt.close()