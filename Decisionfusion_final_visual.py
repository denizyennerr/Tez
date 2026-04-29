import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ⚙️ Configuration
# =============================================================================
# Updated to point to the correct ensemble metrics file
INPUT_CSV = "saved_outputs_play/ensemble_results_final/decision_fusion_macro_mean.csv"
OUTPUT_DIR = os.path.join("saved_outputs_play", "ensemble_results_final", "plots")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global seaborn styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# =============================================================================
# 1. Load Data
# =============================================================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Could not find {INPUT_CSV}. Please ensure the file is in the directory.")

df = pd.read_csv(INPUT_CSV)

# Drop non-metric columns so they don't appear in the plots
columns_to_drop = ['Decision Threshold', 'N_Subjects']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Set the 'Model' column as the index for easier plotting
df_metrics = df.set_index('Model')

# =============================================================================
# 2. Model Performance Heatmap
# =============================================================================
plt.figure(figsize=(12, 6))

# Create Heatmap
sns.heatmap(
    df_metrics,
    annot=True,
    fmt=".4f",
    cmap="RdYlGn", # Red (poor) to Green (excellent)
    cbar_kws={'label': 'Score'},
    vmin=0, vmax=1
)

plt.title("Model vs Evaluation Metrics Heatmap", fontsize=16, fontweight='bold')
plt.ylabel("Model / Ensemble", fontsize=14)
plt.xlabel("Evaluation Metric", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save plot
heatmap_path = os.path.join(OUTPUT_DIR, "model_comparison_heatmap_modelweight.png")
plt.savefig(heatmap_path, dpi=300)
print(f"✅ Heatmap saved to: {heatmap_path}")
plt.close()

# =============================================================================
# 3. Grouped Bar Chart (Metrics Comparison)
# =============================================================================
# Reset index to melt the dataframe for Seaborn's barplot
df_metrics_reset = df_metrics.reset_index()
df_melted = df_metrics_reset.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(15, 7))

# Create Grouped Bar Chart
sns.barplot(
    data=df_melted,
    x='Metric',
    y='Score',
    hue='Model',
    palette='viridis'
)

plt.title("Performance Metrics Comparison Across Models & Ensembles", fontsize=16, fontweight='bold')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Evaluation Metric", fontsize=14)
plt.ylim(0.0, 1.05) # Metrics are between 0 and 1
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Models", borderaxespad=0.)
plt.xticks(rotation=15)
plt.tight_layout()

# Save plot
barchart_path = os.path.join(OUTPUT_DIR, "model_comparison_barchart_weighted.png")
plt.savefig(barchart_path, dpi=300)
print(f"✅ Bar chart saved to: {barchart_path}")
plt.close()