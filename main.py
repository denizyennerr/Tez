import sys
from pathlib import Path
import numpy as np
import os

# Get the path to the 'src' folder and add it to Python's search path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))

# Now this import will work
from eeg import create_multi_scale_windows, CHBMITPreprocessor


def run_pipeline():
    # 1. Define Paths
    base_dir = Path("uvtez/data/chb-mit")
    output_dir = Path("uvtez/data/preprocessed")
    
    if not base_dir.exists():
        print(f"❌ Error: Data directory not found at {base_dir.resolve()}")
        return

    print(f"Input Directory: {base_dir.resolve()}")
    print(f"Output Directory: {output_dir.resolve()}\n")

    # 2. Initialize Preprocessor
    subjects = ['chb01', 'chb03', 'chb05', 'chb09', 'chb10', 
                'chb14', 'chb19', 'chb20', 'chb21', 'chb23']
    
    preprocessor = CHBMITPreprocessor(
        base_dir=str(base_dir),
        output_dir=str(output_dir),
        subject_subset=subjects
    )

    # 3. Process each subject individually
    print("Starting preprocessing...\n")
    for subject in subjects:
        print(f"Processing {subject}...")
        try:
            preprocessor.save_subject(subject)
            print(f" {subject} complete\n")
        except Exception as e:
            print(f" Error processing {subject}: {e}\n")
            continue

    print("All processing complete.\n")

    # 4. Verify output for each subject
    print("--- Verification ---")
    for subject in subjects:
        subject_out_dir = output_dir / subject
        
        if not subject_out_dir.exists():
            print(f" {subject}: Output folder not found")
            continue
            
        generated_files = list(subject_out_dir.glob("*.npz"))
        
        if generated_files:
            first_file = generated_files[0]
            data = np.load(first_file)
            
            if '2s' in data:
                print(f" {subject}: Found '2s' windows in {first_file.name}")
                print(f"   Shape: {data['2s'].shape} (Windows, Channels, Samples)")
            else:
                print(f"  {subject}: '2s' key missing in {first_file.name}")
        else:
            print(f" {subject}: No .npz files generated")
    
if __name__ == "__main__":
    run_pipeline()