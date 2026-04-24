import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ⚙️ Configuration & Academic Styling
# =============================================================================
RESULTS_CSV = "saved_outputs_play\ensemble_results_final\decision_fusion_macro_mean.csv"
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
# 1. Load Data & Set Custom Order for All Models
# =============================================================================
if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Could not find {RESULTS_CSV}. Please ensure it is in the correct directory.")

df = pd.read_csv(RESULTS_CSV)
df = df.drop(columns=['Decision Threshold', 'N_Subjects'], axis=1)

# Define the logical order for ALL models
full_order = [
    "0.5s Individual",
    "1.0s Individual",
    "2.0s Individual",
    "4.0s Individual",
    "5.0s Individual",
    "10.0s Individual",
    "Ensemble (Hard Vote)",
    "Ensemble (Soft Vote)"
]

# Convert 'Model' to an ordered categorical type and sort
df['Model'] = pd.Categorical(df['Model'], categories=full_order, ordered=True)
df_full = df.sort_values('Model').copy()

# =============================================================================
# 2. Overall Performance Heatmap (All Models)
# =============================================================================
plt.figure(figsize=(12, 8))

# Set the Model as the index for the heatmap (it retains the sorted order)
df_heatmap = df_full.set_index("Model")

# Create Heatmap
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

# Note: Consider removing the title if using a document caption.
plt.title("Overall Performance Metrics Across All Models", pad=15)
plt.ylabel("Model")
plt.xlabel("Evaluation Metric")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0) # Ensure y-labels are strictly horizontal

plt.tight_layout()

# Save Heatmap
heatmap_path = os.path.join(OUTPUT_DIR, "all_models_comparison_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✅ Comparison Heatmap saved to: {heatmap_path}")
plt.close()

# =============================================================================
# 3. Focused Grouped Bar Chart (4.0s vs Ensembles)
# =============================================================================
# Define ONLY the models we want to isolate for the bar chart
target_models = [
    "4.0s Individual",
    "Ensemble (Hard Vote)",
    "Ensemble (Soft Vote)"
]

# Filter the dataframe down to just those three
df_filtered = df_full[df_full['Model'].isin(target_models)].copy()

# Remove unused categories from the categorical variable so they don't appear in the legend
df_filtered['Model'] = pd.Categorical(df_filtered['Model'], categories=target_models, ordered=True)

# Melt DataFrame for Seaborn plotting (Grouped Bar Chart format)
df_melted = df_filtered.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 6))

# Use the 'viridis' palette as requested; add edge colors for crisp bar boundaries
ax = sns.barplot(
    data=df_melted,
    x="Metric",
    y="Score",
    hue="Model",
    palette="viridis",
    edgecolor="black",
    linewidth=0.7,
    capsize=0.05
)

plt.title("Performance Comparison: 4.0s Individual vs. Ensemble Models", pad=15)
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

# Save Bar Chart
barplot_path = os.path.join(OUTPUT_DIR, "focused_4s_vs_ensemble_barplot.png")
plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
print(f"✅ Focused Comparison Barplot saved to: {barplot_path}")
plt.close()