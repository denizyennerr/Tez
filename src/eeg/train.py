import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import defaultdict
import warnings

# --- Configuration ---
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
DATA_PATH = project_root / 'data' / 'preprocessed'
PLOTS_DIR = project_root / 'results'

# Hyperparameters
WINDOW_SIZE = '2s'  
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 8
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Loading & Processing ---

def flatten_eeg_window(window):
    """
    Flatten 3D EEG window (samples, channels, time_points) to 2D (samples, features).
    Features = channels * time_points
    """
    n_samples, n_channels, n_timepoints = window.shape
    return window.reshape(n_samples, n_channels * n_timepoints)

def load_patient_data(patient_ids=['chb01', 'chb03', 'chb05'], window_size='2s', exclude_boundary=True):
    """
    Load and aggregate EEG data for a specific patient.
    
    Args:
        patient_ids: List of patient folder names (e.g., ['chb01'])
        window_size: Which window size to use ('1s', '2s', '5s', '8s', '10s')
        exclude_boundary: If True, excludes samples with label=0 (boundary/transition)
    """
    # Handle single string input
    if isinstance(patient_ids, str):
        patient_ids = [patient_ids]

    all_files = []
    
    # 1. Collect files from all patient directories
    print(f"Scanning directories for patients: {patient_ids}")
    for p_id in patient_ids:
        patient_dir = DATA_PATH / p_id
        
        if not patient_dir.exists():
            print(f"  Warning: Directory not found for {p_id}, skipping...")
            continue

        files = sorted(list(patient_dir.glob("*.npz")))
        if not files:
            print(f"  Warning: No .npz files found in {patient_dir}, skipping...")
            continue
            
        all_files.extend(files)

    if not all_files:
        raise FileNotFoundError("No files found for the provided patient IDs.")

    print(f"Loading {len(all_files)} files total (window size: {window_size})...")
    
    X_list = []
    y_list = []
    file_id_list = []

    # Process all files
    for file_idx, f in enumerate(tqdm(all_files, desc="Reading files")):
        try:
            data = np.load(f)
            
            # Validate keys
            if window_size not in data.keys():
                # Fallback check
                available = [k for k in data.keys() if k != 'labels']
                print(f"Warning: {window_size} not found in {f.name}. Available: {available}")
                continue
                
            if 'labels' not in data.keys():
                raise KeyError(f"'labels' key not found in {f.name}")
            
            # Load data and labels
            X_window = data[window_size]  # Shape: (samples, channels, time_points)
            y_labels = data['labels']      # Shape: (samples,)
            
            # Flatten to 2D
            X_flat = flatten_eeg_window(X_window)
            
            # Filter out boundary labels if requested
            if exclude_boundary:
                valid_mask = y_labels != 0
                X_flat = X_flat[valid_mask]
                y_labels = y_labels[valid_mask]
            
            # Skip empty files
            if len(y_labels) == 0:
                continue
            
            # Create file IDs for each sample
            file_ids = np.full(len(y_labels), file_idx)
            
            X_list.append(X_flat)
            y_list.append(y_labels)
            file_id_list.append(file_ids)
            
        except Exception as e:
            print(f"ERROR loading {f.name}: {e}")
            continue # Skip bad files instead of crashing

    if not X_list:
        raise ValueError("No valid data loaded.")

    # Concatenate all sessions
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    file_ids = np.concatenate(file_id_list, axis=0)
    
    # Convert labels: -1 -> 0 (non-seizure), 1 -> 1 (seizure)
    y_binary = (y == 1).astype(np.float32)
    
    print(f"Total Data Shape: X={X.shape}, y={y_binary.shape}")
    print(f"Class Distribution:")
    print(f"  Non-Seizure (0): {np.sum(y_binary == 0)} ({np.sum(y_binary == 0)/len(y_binary)*100:.2f}%)")
    print(f"  Seizure (1): {np.sum(y_binary == 1)} ({np.sum(y_binary == 1)/len(y_binary)*100:.2f}%)")
    
    return X, y_binary, file_ids

def file_based_split(X, y, file_ids, test_ratio=0.2):
    """
    Split data by files (not samples) to prevent temporal leakage.
    Ensures entire recording files are in either train or test, never both.
    """
    unique_files = np.unique(file_ids)
    n_files = len(unique_files)
    n_test = max(1, int(n_files * test_ratio))
    
    # Shuffle files
    np.random.shuffle(unique_files)
    test_files = unique_files[:n_test]
    train_files = unique_files[n_test:]
    
    # Create masks
    train_mask = np.isin(file_ids, train_files)
    test_mask = np.isin(file_ids, test_files)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"\nFile-based split:")
    print(f"  Train files: {len(train_files)}, Test files: {len(test_files)}")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def undersample_data(X, y, random_state=None):
    """
    Undersamples the majority class (Non-Seizure/0) to match the minority class (Seizure/1).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    print("\nApplying undersampling to training data...")
    
    # Ensure y is 1D
    if y.ndim > 1: 
        y = y.squeeze()

    seizure_indices = np.where(y == 1)[0]
    non_seizure_indices = np.where(y == 0)[0]
    
    n_minority = len(seizure_indices)
    n_majority = len(non_seizure_indices)
    
    if n_minority == 0:
        print("WARNING: No seizure samples found in training data! Skipping undersampling.")
        return X, y

    print(f"  Before: Non-Seizure={n_majority}, Seizure={n_minority} (ratio: {n_majority/n_minority:.1f}:1)")
    
    if n_majority > n_minority:
        # Randomly sample majority class
        undersampled_non = np.random.choice(non_seizure_indices, size=n_minority, replace=False)
    else:
        undersampled_non = non_seizure_indices

    # Combine and shuffle
    balanced_indices = np.concatenate([seizure_indices, undersampled_non])
    np.random.shuffle(balanced_indices)
    
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    print(f"  After: Non-Seizure={np.sum(y_balanced==0)}, Seizure={np.sum(y_balanced==1)} (ratio: 1:1)")
    
    return X_balanced, y_balanced

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 2. Optimized KAN Architecture ---

class VectorizedKAN(nn.Module):
    """
    Optimized KAN using Grouped Convolutions for EEG seizure detection.
    """
    def __init__(self, input_size, hidden_size=8):
        super(VectorizedKAN, self).__init__()
        
        self.input_transform = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size * hidden_size,
            kernel_size=1,
            groups=input_size
        )

        self.activation = nn.SiLU()

        self.output_transform = nn.Conv1d(
            in_channels=input_size * hidden_size,
            out_channels=input_size,
            kernel_size=1,
            groups=input_size
        )

        self.combination = nn.Linear(input_size, 1)

    def forward(self, x):
        # x shape: (Batch, Features)
        x_reshaped = x.unsqueeze(2)  # (Batch, Features, 1)
        
        out = self.input_transform(x_reshaped)
        out = self.activation(out)
        out = self.output_transform(out)
        
        out = out.squeeze(2)  # (Batch, Features)
        out = self.combination(out)
        return out

# --- 3. Training & Evaluation ---

def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    
    # Calculate Metrics
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0)
    }
    return metrics, all_preds, all_targets

def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    history = defaultdict(list)
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation
        val_metrics, _, _ = evaluate_model(model, test_loader, criterion)
        history['test_loss'].append(val_metrics['loss'])
        history['accuracy'].append(val_metrics['accuracy'])
        history['precision'].append(val_metrics['precision'])
        history['recall'].append(val_metrics['recall'])
        history['f1'].append(val_metrics['f1'])
        
        if (epoch+1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {epoch_train_loss:.4f} | "
                  f"Test Loss: {val_metrics['loss']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f}")
            
    return history

def plot_results(history, save_path):
    """Create comprehensive training visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['test_loss'], label='Test Loss', color='cyan')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 1].plot(history['f1'], label='F1 Score', color='green')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history['precision'], label='Precision', color='orange')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history['recall'], label='Recall', color='red')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-Seizure', 'Seizure']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")

# --- Main Execution ---

if __name__ == "__main__":
    try:
        # 1. Load Real Data with proper window size
        X_raw, y_raw, file_ids = load_patient_data(
            patient_ids=['chb01', 'chb03', 'chb05'], 
            window_size=WINDOW_SIZE,
            exclude_boundary=True
        )
        
        # 2. File-based split to prevent temporal leakage
        X_train_raw, X_test, y_train_raw, y_test = file_based_split(
            X_raw, y_raw, file_ids, test_ratio=TEST_SPLIT
        )
        
        # 3. Apply Undersampling to training set ONLY
        X_train, y_train = undersample_data(X_train_raw, y_train_raw, random_state=RANDOM_SEED)
        
        # 4. Prepare Loaders
        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 5. Initialize Model
        input_dim = X_train.shape[1]
        print(f"\nInitializing Vectorized KAN:")
        print(f"  Input Dimension: {input_dim}")
        print(f"  Hidden Size: {HIDDEN_SIZE}")
        
        model = VectorizedKAN(input_size=input_dim, hidden_size=HIDDEN_SIZE).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total Parameters: {total_params:,}")
        
        # 6. Train
        history = train_model(
            model, train_loader, test_loader, 
            num_epochs=NUM_EPOCHS, 
            learning_rate=LEARNING_RATE
        )
        
        # 7. Final Evaluation
        print("\n" + "="*50)
        print("FINAL EVALUATION")
        print("="*50)
        
        criterion = nn.BCEWithLogitsLoss()
        final_metrics, y_pred, y_true = evaluate_model(model, test_loader, criterion)
        
        print(f"Test Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"Test Precision: {final_metrics['precision']:.4f}")
        print(f"Test Recall:    {final_metrics['recall']:.4f}")
        print(f"Test F1 Score:  {final_metrics['f1']:.4f}")
        
        # 8. Visualizations
        plot_results(history, PLOTS_DIR / 'kan_training_results.png')
        plot_confusion_matrix(y_true, y_pred, PLOTS_DIR / 'confusion_matrix.png')
        
        # 9. Save Model
        model_save_path = project_root / 'results'
        model_save_path.mkdir(exist_ok=True, parents=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': dict(history),
            'config': {
                'window_size': WINDOW_SIZE,
                'input_dim': input_dim,
                'hidden_size': HIDDEN_SIZE,
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'learning_rate': LEARNING_RATE
            }
        }, model_save_path / 'kan_model.pth')
        
        print(f"\nModel and config saved to {model_save_path / 'kan_model.pth'}")
        print("Training complete!")
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*50}")