import sys
from pathlib import Path

# Ensure the src folder is in the python path so we can import the module
# Assuming this script is running from the 'uvtez' root directory
sys.path.append(str(Path.cwd() / 'src'))

# Import your class (adjust import path based on your exact file structure)
# If your file is at src/eeg/preprocessing.py:
from eeg.preprocessing import CHBMITPreprocessor

def run_pipeline():
    # 1. Define Paths based on your screenshot structure
    base_dir = Path("uvtez/data/chb-mit")
    output_dir = Path("uvtez/data/preprocessed")
    
    # Ensure directories exist
    if not base_dir.exists():
        print(f"Error: Data directory not found at {base_dir.resolve()}")
        return

    print(f"Input Directory: {base_dir}")
    print(f"Output Directory: {output_dir}")

    # 2. Initialize the Preprocessor
    # We stick to defaults, which include the 2.0s window in WINDOW_LENGTHS
    preprocessor = CHBMITPreprocessor(
        base_dir=str(base_dir),
        output_dir=str(output_dir),
        subject_subset=['chb01', 'chb03', 'chb05', 'chb09', 'chb10', 'chb14', 'chb19', 'chb20', 'chb21', 'chb23'] 
    )

    # 3. Run processing for Subject 01
    print("\nStarting preprocessing for chb01...")
    preprocessor.save_subject('chb01', 'chb03', 'chb05', 'chb09', 'chb10', 'chb14', 'chb19', 'chb20', 'chb21', 'chb23')
    print("\n✅ Processing complete.")

    # 4. Verify the 2s window data exists in the output
    print("\n--- Verification ---")
    subject_out_dir = output_dir / "chb01" / "chb03" / "chb05" / "chb09" / "chb10" / "chb14" / "chb19" / "chb20" / "chb21" / "chb23"
    
    # Check the first generated file
    generated_files = list(subject_out_dir.glob("*.npz"))
    if generated_files:
        import numpy as np
        first_file = generated_files[0]
        data = np.load(first_file)
        
        if '2s' in data:
            print(f"Successfully verified '2s' window data in {first_file.name}")
            print(f"Shape of 2s data: {data['2s'].shape} (Windows, Channels, Samples)")
            print(f"Shape of labels: {data['labels'].shape}")
        else:
            print(f"⚠️ '2s' key not found in {first_file.name}. Check WINDOW_LENGTHS in class.")
    else:
        print("No output files were generated.")

if __name__ == "__main__":
    run_pipeline()
