import numpy as np
import pandas as pd

# 1. Load the 0.5s dataset
data05_path = "processed_master_datasets/master_dataset_10.0s.npz"
data05 = np.load(data05_path)

# 2. Extract arrays (ensure the keys match how you saved them)
# X = data05["X"] # You don't strictly need X for this summary
y = data05["y"]  # Assuming 1 = seizure, 0 = normal
patient_ids = data05["s"]  # Assuming you saved an array of patient IDs

# 3. Create a DataFrame to make grouping easier
df = pd.DataFrame({
    'patient_id': patient_ids,
    'label': y
})

# 4. Group by patient_id and calculate the metrics
summary_list = []

for patient, group in df.groupby('patient_id'):
    total_windows = len(group)
    seizure_windows = int(group['label'].sum())
    normal_windows = total_windows - seizure_windows
    seizure_percentage = round((seizure_windows / total_windows) * 100, 2)

    summary_list.append({
        'patient_id': patient,
        'seizure_windows': seizure_windows,
        'normal_windows': normal_windows,
        'total_windows': total_windows,
        'seizure_percentage': seizure_percentage
    })

# 5. Convert to DataFrame and save to CSV
summary_df = pd.DataFrame(summary_list)

output_csv = "final_dataset_all_patients_10.0s.csv"
summary_df.to_csv(output_csv, index=False)

print(f"Summary dataset saved to {output_csv}")
print(summary_df.head())