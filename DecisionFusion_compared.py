import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ⚙️ Configuration
# =============================================================================
# Map the display names to their respective summary CSV files.
# Update the paths to the individual 1s, 2s, 4s, 5s, and 10s CSVs as needed.
MODEL_CSVS = {
    "1s Model": "saved_outputs/20260304-135510-1s/overall_loso_summary_1s.csv",
    "2s Model": "saved_outputs/20260303-162053-2s/overall_loso_summary_2s.csv",
    "4s Model": "saved_outputs/20260304-091718-4s/overall_loso_summary_4s.csv",
    "Ensemble (Opt Thresh Soft)": "saved_outputs/ensemble_results_static/ensemble_opt_thresh_soft_summary.csv",

    # Optional: If you also ran the 'max' fusion and saved it to the new static folder, you can include it here:
    # "Ensemble (Max)":           "saved_outputs/ensemble_results_static/ensemble_max_summary.csv"
}

OUTPUT_DIR = os.path.join("saved_outputs", "ensemble_results_static", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global seaborn styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# =============================================================================
# 1. Load and Aggregate Data
# =============================================================================
metrics_of_interest = [
    "auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "seizure_f1"
]

metric_names_mapping = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced Accuracy",
    "seizure_f1": "Seizure F1-Score"
}

aggregated_data = []

for model_name, csv_path in MODEL_CSVS.items():
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: Could not find {csv_path}. Skipping {model_name}...")
        continue

    df = pd.read_csv(csv_path)

    # Check if this is a per-subject file that needs averaging,
    # or an already averaged summary file.
    if 'subject' in df.columns:
        df_mean = df[metrics_of_interest].mean().to_dict()
    else:
        # Assumes it's already an aggregated row
        df_mean = df[metrics_of_interest].iloc[0].to_dict()

    df_mean['Model'] = model_name
    aggregated_data.append(df_mean)

if not aggregated_data:
    raise ValueError("No valid CSV files were found. Please check your paths.")

# Create the combined DataFrame
df_combined = pd.DataFrame(aggregated_data)
df_combined.rename(columns=metric_names_mapping, inplace=True)

# Save the aggregated summary table for your records
summary_table_path = os.path.join(OUTPUT_DIR, "all_models_performance_comparison.csv")
df_combined.to_csv(summary_table_path, index=False)
print(f"✅ Saved aggregated comparison table to: {summary_table_path}")

# =============================================================================
# 2. Comparative Bar Chart (Grouped by Metric)
# =============================================================================
# Melt DataFrame for Seaborn grouped barplot
df_melted = df_combined.melt(id_vars=["Model"], var_name="Evaluation Metric", value_name="Score")

plt.figure(figsize=(16, 8))

sns.barplot(
    data=df_melted,
    x="Evaluation Metric",
    y="Score",
    hue="Model",
    palette="Set2"
)

plt.title("Performance Comparison Across All Epoched & Ensemble Models", fontsize=18, fontweight='bold')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Evaluation Metric", fontsize=14)
plt.ylim(0.0, 1.05)
plt.xticks(rotation=15)
plt.legend(title="Model Type", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()

barplot_path = os.path.join(OUTPUT_DIR, "all_models_comparison_barplot.png")
plt.savefig(barplot_path, dpi=300, bbox_inches="tight")
print(f"✅ Bar chart saved to: {barplot_path}")
plt.close()

# =============================================================================
# 3. Overall Performance Heatmap (Models vs Metrics)
# =============================================================================
plt.figure(figsize=(12, 8))

# Set Model as index for the heatmap
df_heatmap = df_combined.set_index("Model")

sns.heatmap(
    df_heatmap,
    annot=True,
    fmt=".4f",
    cmap="RdYlGn",  # Red (poor) to Green (excellent)
    cbar_kws={'label': 'Score'},
    vmin=0.2, vmax=1.0  # Adjusted vmin to show better contrast based on typical scores
)

plt.title("Overall Performance Metrics Across All Models", fontsize=18, fontweight='bold')
plt.ylabel("Model", fontsize=14)
plt.xlabel("Metrics", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, "all_models_comparison_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
print(f"✅ Heatmap saved to: {heatmap_path}")
plt.close()