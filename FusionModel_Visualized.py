import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ⚙️ Configuration
# =============================================================================
VOTING_METHOD = 'hard'
RESULTS_CSV = os.path.join("saved_outputs", "ensemble_results", f"ensemble_{VOTING_METHOD}_summary.csv")
OUTPUT_DIR = os.path.join("saved_outputs", "ensemble_results", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global seaborn styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# =============================================================================
# 1. Load Data
# =============================================================================
if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Could not find {RESULTS_CSV}. Please run the ensemble evaluation first.")

df = pd.read_csv(RESULTS_CSV)

# Select metrics of interest
metrics_of_interest = [
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "seizure_f1"
]

# Rename columns for prettier plot labels
metric_names_mapping = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced Accuracy",
    "seizure_f1": "Seizure F1-Score"
}

df_metrics = df[['subject'] + metrics_of_interest].copy()
df_metrics.rename(columns=metric_names_mapping, inplace=True)

# Melt DataFrame for Seaborn plotting
df_melted = df_metrics.melt(id_vars=["subject"], var_name="Metric", value_name="Score")

# =============================================================================
# 2. Boxplot (Metric Distributions)
# =============================================================================
plt.figure(figsize=(12, 6))

# Boxplot for the statistical summary
ax = sns.boxplot(
    data=df_melted,
    x="Metric",
    y="Score",
    palette="pastel",
    showfliers=True # Re-enabled so you can still see statistical outliers
)

plt.title(f"Ensemble Model Performance Distribution across Subjects ({VOTING_METHOD.capitalize()} Voting)", fontsize=16, fontweight='bold')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Evaluation Metric", fontsize=14)
plt.ylim(0.0, 1.05) # Metrics are mostly between 0 and 1
plt.xticks(rotation=15)
plt.tight_layout()

# Save plot
boxplot_path = os.path.join(OUTPUT_DIR, f"metrics_distribution_{VOTING_METHOD}.png")
plt.savefig(boxplot_path, dpi=300)
print(f"✅ Boxplot saved to: {boxplot_path}")
plt.close()

# =============================================================================
# 3. Subject-wise Performance Heatmap
# =============================================================================
plt.figure(figsize=(10, 8))

# Set the subject as index
df_heatmap = df_metrics.set_index("subject")

# Sort index alphanumerically to ensure chb01-chb24 ordering
df_heatmap = df_heatmap.sort_index()

# Create Heatmap
sns.heatmap(
    df_heatmap,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn", # Red (poor) to Green (excellent)
    cbar_kws={'label': 'Score'},
    vmin=0, vmax=1
)

plt.title(f"Subject-wise Evaluation Metrics ({VOTING_METHOD.capitalize()} Voting)", fontsize=16, fontweight='bold')
plt.ylabel("Subject ID", fontsize=14)
plt.xlabel("Metrics", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save plot
heatmap_path = os.path.join(OUTPUT_DIR, f"subject_heatmap_{VOTING_METHOD}.png")
plt.savefig(heatmap_path, dpi=300)
print(f"✅ Heatmap saved to: {heatmap_path}")
plt.close()