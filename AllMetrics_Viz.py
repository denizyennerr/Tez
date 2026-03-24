import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ⚙️ Configuration
# =============================================================================
RESULTS_CSV = "saved_outputs_play/ensemble_results_rigorous/decision_fusion_macro_mean.csv"
OUTPUT_DIR = os.path.join("saved_outputs_play", "ensemble_results_rigorous", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global seaborn styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# =============================================================================
# 1. Load Data
# =============================================================================
if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Could not find {RESULTS_CSV}. Please ensure it is in the correct directory.")

df = pd.read_csv(RESULTS_CSV)
df= df.drop(columns=['Decision Threshold','N_Subjects'], axis=1)

# Melt DataFrame for Seaborn plotting (Grouped Bar Chart format)
df_melted = df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

# =============================================================================
# 2. Grouped Bar Chart (All Models Comparison)
# =============================================================================
plt.figure(figsize=(16, 8))

# Grouped barplot: X-axis = Metric, Y-axis = Score, Color = Model
ax = sns.barplot(
    data=df_melted,
    x="Metric",
    y="Score",
    hue="Model",
    palette="Set2"  # Distinct color palette for different models
)

plt.title("Performance Comparison Across All Epoched & Ensemble Models", fontsize=18, fontweight='bold')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Evaluation Metric", fontsize=14)
plt.ylim(0.0, 1.05) # Metrics are between 0 and 1
plt.xticks(rotation=15)

# Place the legend outside the plot so it doesn't overlap the bars
plt.legend(title="Model Type", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
plt.tight_layout()

# Save plot
barplot_path = os.path.join(OUTPUT_DIR, "all_models_comparison_barplot.png")
plt.savefig(barplot_path, dpi=300)
print(f"✅ Comparison Barplot saved to: {barplot_path}")
plt.close()

# =============================================================================
# 3. Overall Performance Heatmap
# =============================================================================
plt.figure(figsize=(12, 8))

# Set the Model as the index for the heatmap
df_heatmap = df.set_index("Model")

# Create Heatmap
sns.heatmap(
    df_heatmap,
    annot=True,
    fmt=".4f",
    cmap="RdYlGn", # Red (poor) to Green (excellent)
    cbar_kws={'label': 'Score'},
    vmin=0.2, vmax=1.0 # Adjusted minimum scale so differences pop out more
)

plt.title("Overall Performance Metrics Across All Models", fontsize=16, fontweight='bold')
plt.ylabel("Model", fontsize=14)
plt.xlabel("Metrics", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save plot
heatmap_path = os.path.join(OUTPUT_DIR, "all_models_comparison_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
print(f"✅ Comparison Heatmap saved to: {heatmap_path}")
plt.close()