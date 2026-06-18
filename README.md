# Multi-Resolution Deep Learning Framework for Seizure Detection in Multi-Channel EEG

This repository contains the codebase for a deep learning pipeline designed to predict and detect epileptic seizures from raw multi-channel EEG signals, using the **CHB-MIT Scalp EEG Database**.

## 🛠 Environment Setup

To ensure all scripts run correctly, create a dedicated conda environment with all the necessary dependencies.

```bash
# 1. Create the environment
conda create -n tez python=3.10 -y

# 2. Activate the environment
conda activate tez

# 3. Install core packages using conda
conda install -c conda-forge numpy pandas scipy matplotlib seaborn scikit-learn ipython jupyter -y

# 4. Install ML libraries and MNE using pip
pip install tensorflow mne
```

---

## 🚀 Pipeline Workflow

To reproduce the workflow from raw data to final evaluation, follow these steps in order:

### 1. Data Preprocessing & Generation
**Script:** `dataset_create_master_all.py`

This script reads the raw EDF files from the CHB-MIT dataset (expected to be in `data-understanding/data/chb-mit`), preprocesses the signals using the `EEGPreprocessor.py` pipeline (bandpass, notch filter, downsampling, z-score normalization), and generates smaller `.npz` files for each subject into the directory `processed_master_datasets/`. 

*Note: You can change the `WINDOW_SIZE` variable inside the script to extract data for different resolutions (e.g., 0.5s, 1.0s, 2.0s).*

```bash
python dataset_create_master_all.py
```

### 2. Dataset Aggregation
**Script:** `loso_dataset_creation.py`

Once all individual `.npz` files are created, this script merges them into a single, unified master dataset (e.g., `master_dataset_0.5s.npz`). This format is optimized for Leave-One-Subject-Out (LOSO) cross-validation training.

```bash
python loso_dataset_creation.py
```

### 3. Model Training
**Script:** `model_training.py` or `Loso_model_training.py`

Using the aggregated `.npz` file, this step trains the deep learning model (defined in `model.py` or `CNN-LSTM-Attention.py`). The training uses Leave-One-Subject-Out (LOSO) cross-validation to ensure models generalize well to unseen patients. Checkpoints, models, and training logs will be saved to disk.

```bash
python model_training.py
```

### 4. Evaluation & Visualization
**Scripts:** `model_eval_updated.py`, `TestingResults_Aggregate.py`, `AllMetrics_visualized.py`

These scripts calculate comprehensive metrics (Sensitivity, Specificity, AUC, F1-Score) on the test subjects and generate aggregate visualizations.

```bash
python model_eval_updated.py
```

### 5. Multi-Resolution / Decision Fusion
**Scripts:** `ensemble_learning_final.py`, `decision_fusion_final.py`

Combines predictions from multiple models trained on different window sizes (e.g., 0.5s + 1.0s + 2.0s). The decision fusion aggregates the probabilities to yield a more robust and accurate final seizure detection classification.

---

## 📁 Repository Structure

* `EEGPreprocessor.py` — The core preprocessing class handling sampling, filtering, and normalization.
* `model.py` / `CNN-LSTM-Attention.py` — Neural Network architectures.
* `handy/` — Contains utility scripts like `preprocess_utility.py` (which uses MNE for EDF parsing).
* `data-understanding/` — Directory containing exploratory Jupyter notebooks (`preprocessing.ipynb`, `data_analysis.ipynb`) and raw datasets.
