import os
import glob
import pandas as pd

# =============================================================================
# ⚙️ Configuration
# =============================================================================
BASE_DIR = "saved_outputs"

# The specific metrics we want to extract and compare
METRICS = [
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "seizure_f1"
]

# Rename columns for the final presentation table
METRIC_NAMES = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced Accuracy",
    "seizure_f1": "Seizure F1-Score"
}


def aggregate_all_results():
    print(f"📁 Searching for summary CSVs in '{BASE_DIR}'...\n")

    # 1. Find all individual model summary CSVs recursively
    individual_files = glob.glob(os.path.join(BASE_DIR, "**", "overall_loso_summary_*.csv"), recursive=True)

    # 2. Find all ensemble summary CSVs
    ensemble_files = glob.glob(os.path.join(BASE_DIR, "**", "ensemble_*_summary.csv"), recursive=True)

    all_files = individual_files + ensemble_files
    results_list = []

    if not all_files:
        print("⚠️ No summary CSV files found. Please check your directory structure.")
        return

    # 3. Process each file
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)

            # Determine the model name from the filename
            filename = os.path.basename(file_path)
            if "ensemble" in filename:
                # Extracts 'hard' or 'soft' from 'ensemble_hard_summary.csv'
                voting_type = filename.split('_')[1]
                model_name = f"Ensemble ({voting_type.capitalize()})"
            else:
                # Extracts '10s' from 'overall_loso_summary_10s.csv'
                suffix = filename.replace(".csv", "").split("_")[-1]
                model_name = f"{suffix} Model"

            # Check if required metrics exist in the DataFrame
            missing_metrics = [m for m in METRICS if m not in df.columns]
            if missing_metrics:
                print(f"⚠️ Skipping {model_name} because missing columns: {missing_metrics}")
                continue

            # Calculate the mean across all subjects for this model
            means = df[METRICS].mean().to_dict()

            # Format the row for our final table
            row = {"Model": model_name}
            for k, v in means.items():
                row[METRIC_NAMES[k]] = round(v, 4)

            results_list.append(row)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    # 4. Generate the final table
    if results_list:
        final_df = pd.DataFrame(results_list)

        # Define a custom sort so epoch numbers appear in order, with Ensemble at the bottom
        def sort_key(model_name):
            if "Ensemble" in model_name:
                return float('inf')  # Push ensemble to the end
            else:
                # Extract the number from '10s Model' for sorting
                num_str = ''.join([c for c in model_name if c.isdigit() or c == '.'])
                return float(num_str) if num_str else 999

        # Apply sorting
        final_df['sort_val'] = final_df['Model'].apply(sort_key)
        final_df = final_df.sort_values('sort_val').drop('sort_val', axis=1).reset_index(drop=True)

        # 5. Display and Save
        print("=" * 85)
        print("🏆 Aggregated Model Performance Comparison (Mean across Subjects) 🏆")
        print("=" * 85)
        print(final_df.to_string(index=False))
        print("=" * 85)

        # Save to the root of saved_outputs
        out_csv = os.path.join(BASE_DIR, "all_models_performance_comparison.csv")
        final_df.to_csv(out_csv, index=False)
        print(f"\n✅ Comparison table saved to: {out_csv}")


if __name__ == "__main__":
    aggregate_all_results()