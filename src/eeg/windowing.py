import sys
import numpy as np
from pathlib import Path

# --- Setup Paths ---
# Ensure we are pointing to the correct root
# If you run this file from 'uvtez/', this works.
ROOT_DIR = Path.cwd()
sys.path.append(str(ROOT_DIR / 'src'))

try:
    from eeg.preprocessing import CHBMITPreprocessor
except ImportError:
    print("❌ Error: Could not import CHBMITPreprocessor. Check your 'src' folder structure.")
    sys.exit(1)

def run_pipeline():
    # 1. Define Paths
    base_dir = ROOT_DIR / "data/chb-mit"
    output_dir = ROOT_DIR / "data/preprocessed"
    
    # 2. Define your subjects
    subjects_to_process = [
        'chb01', 'chb03', 'chb05', 'chb09', 'chb10', 
        'chb14', 'chb19', 'chb20', 'chb21', 'chb23'
    ]

    print(f"Input Directory: {base_dir}")
    print(f"Output Directory: {output_dir}")

    # 3. Initialize Preprocessor
    # We pass the subset here, but we still need to loop through them below
    preprocessor = CHBMITPreprocessor(
        base_dir=str(base_dir),
        output_dir=str(output_dir),
        subject_subset=subjects_to_process
    )
    
    # Ensure 2.0s is in the window lengths
    if 2.0 not in preprocessor.WINDOW_LENGTHS:
        print("⚠️ Warning: 2.0s window not found in preprocessor defaults!")

    # 4. Run Processing Loop (FIXED)
    print(f"\n🚀 Starting preprocessing for {len(subjects_to_process)} subjects...")
    
    for subject_id in subjects_to_process:
        print(f"\nProcessing {subject_id}...")
        try:
            # We call save_subject for ONE patient at a time
            preprocessor.save_subject(subject_id)
        except Exception as e:
            print(f"❌ Failed to process {subject_id}: {e}")

    print("\n✅ All processing attempts complete.")

    # 5. Verify '2s' Data (FIXED)
    print("\n🔎 --- Verification ---")
    
    # We check the folder for the FIRST subject only (chb01)
    test_subject = subjects_to_process[0]
    subject_out_dir = output_dir / test_subject
    
    # Check if files exist
    generated_files = list(subject_out_dir.glob("*.npz"))
    
    if generated_files:
        first_file = generated_files[0]
        print(f"Inspecting file: {first_file.name}")
        
        try:
            data = np.load(first_file)
            print(f"Keys found in file: {list(data.keys())}")
            
            # Specifically check for 2s window
            if '2s' in data:
                print(f"✅ SUCCESS: '2s' window data found.")
                print(f"   Shape: {data['2s'].shape} (Windows, Channels, Samples)")
                print(f"   Labels: {data['labels'].shape}")
            else:
                print(f"❌ ERROR: '2s' key NOT found. Check WINDOW_LENGTHS in preprocessing.py")
                
        except Exception as e:
            print(f"❌ Error reading .npz file: {e}")
    else:
        print(f"❌ No output files found in {subject_out_dir}")

if __name__ == "__main__":
    run_pipeline()