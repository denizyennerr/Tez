import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from eeg.models import KANSeizureDetector

class NPZDataset(Dataset):
    """
    Lazy-loading Dataset for preprocessed NPZ files.
    
    This class loads data samples on-demand rather than loading the entire
    file into memory at once, making it memory-efficient for large datasets.
    """
    def __init__(self, npz_path: Path, window_size_key: str = '2s'):
        self.npz_path = npz_path
        self.window_size_key = window_size_key
        
        # Load only metadata, not the actual data
        try:
            with np.load(npz_path, allow_pickle=True) as npz_file:
                available_keys = list(npz_file.keys())
                
                # Check which key contains the data
                if window_size_key in npz_file:
                    self.data_key = window_size_key
                    self.n_samples = len(npz_file[window_size_key])
                elif 'data' in npz_file:
                    self.data_key = 'data'
                    self.n_samples = len(npz_file['data'])
                else:
                    raise KeyError(
                        f"Data key not found in {npz_path.name}.\n"
                        f"  Looking for: '{window_size_key}' or 'data'\n"
                        f"  Available keys: {available_keys}"
                    )
                
                # Verify labels exist
                if 'labels' not in npz_file:
                    raise KeyError(
                        f"'labels' not found in {npz_path.name}.\n"
                        f"  Available keys: {available_keys}"
                    )
                    
                # Store data shape for verification
                self.data_shape = npz_file[self.data_key].shape
                
        except Exception as e:
            raise RuntimeError(f"Error initializing NPZDataset for {npz_path.name}: {e}")
   
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.n_samples})")
        
        # Load data on demand (lazy loading)
        try:
            with np.load(self.npz_path, allow_pickle=True) as npz_file:
                data = npz_file[self.data_key][idx]
                label = npz_file['labels'][idx]
        except Exception as e:
            raise RuntimeError(
                f"Error loading sample {idx} from {self.npz_path.name}: {e}"
            )
        
        # Convert to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # Flatten if necessary (Time, Channels) -> (Features)
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.flatten()
        elif data_tensor.dim() > 2:
            # Flatten all dimensions
            data_tensor = data_tensor.flatten()
        
        # Ensure label is 1D with single element
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)
        elif label_tensor.numel() > 1:
            # If multi-element, take the first one (or handle as needed)
            label_tensor = label_tensor.flatten()[0:1]
        
        return data_tensor, label_tensor


# Get current working directory
current_dir = Path(os.getcwd()).resolve()

if 'uvtez' in str(current_dir):
    # Find the part of the path in 'uvtez'
    while current_dir.name != 'uvtez' and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    project_root = current_dir
else:
    project_root = Path(os.getcwd()).resolve()

# Define paths
src_path = project_root / 'src'
DATA_PATH = project_root / 'data' / 'preprocessed'  

# Add paths to sys.path
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

print(f" Project Root: {project_root}")
print(f" Data Path:    {DATA_PATH}")

# Import model after path setup
try:
    from src.eeg.models import KANSeizureDetector
    print(" Success! Imported via 'src.eeg.models'")
except ImportError:
    try:
        from eeg.models import KANSeizureDetector
        print(" Success! Imported via 'eeg.models'")
    except ImportError as e:
        print(f" Failed to import model. Error: {e}")
        sys.exit(1) 

# HYPERPARAMETERS
SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
GRID_SIZE = 5
DROPOUT = 0.2
WINDOW_SIZE_KEY = '2s'
L1_LAMBDA = 1e-5
ENTROPY_LAMBDA = 1e-5

# SUBJECT-LEVEL CROSS-VALIDATION SPLIT
N_TRAIN = 3 
N_VAL = 1  
N_TEST = 1  

# Setting the seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# LOAD DATA FUNCTION - Returns file paths organized by subject
def load_data(data_path: Path, window_size_key: str) -> Dict[str, List[Path]]:
    """
    Load file paths from the preprocessed directory, organized by subject.
    Does NOT load actual data into memory - returns paths for lazy loading.
    """
    data_by_subject = {}
    skipped_files = []
    
    for file_path in data_path.glob("**/*.npz"):
        try:
            # Extract subject ID from filename (e.g., chb01_01.npz -> chb01)
            subject_id = file_path.stem.split('_')[0]
            
            # Verify the file contains required keys
            with np.load(file_path) as data:
                has_data = window_size_key in data or 'data' in data
                has_labels = 'labels' in data
                
                if has_data and has_labels:
                    if subject_id not in data_by_subject:
                        data_by_subject[subject_id] = []
                    data_by_subject[subject_id].append(file_path)
                    print(f" ✓ Registered {file_path.name} for subject {subject_id}")
                else:
                    missing = []
                    if not has_data:
                        missing.append(f"'{window_size_key}' or 'data'")
                    if not has_labels:
                        missing.append("'labels'")
                    skipped_files.append((file_path.name, ', '.join(missing)))
                
        except (OSError, KeyError) as e:
            print(f" ✗ Error checking {file_path.name}: {e}")
    
    if skipped_files:
        print(f"\n⚠ Skipped {len(skipped_files)} files:")
        for fname, reason in skipped_files[:5]:  # Show first 5
            print(f"  - {fname}: Missing {reason}")
        if len(skipped_files) > 5:
            print(f"  ... and {len(skipped_files) - 5} more")
    
    return data_by_subject

# Create datasets from subject splits using lazy loading
def create_dataset_from_subjects(
    subjects: List[str], 
    data_dict: Dict[str, List[Path]], 
    window_size_key: str
) -> Optional[ConcatDataset]:
    """
    Creates a ConcatDataset from NPZ files for specific subjects.
    Uses lazy loading - data is only loaded when accessed.
    
    Args:
        subjects: List of subject IDs
        data_dict: Dictionary mapping subject IDs to lists of NPZ file paths
        window_size_key: Key for the windowed data in NPZ files
        
    Returns:
        ConcatDataset of NPZDataset objects, or None if no subjects provided
    """
    if not subjects:
        return None
    
    if not data_dict:
        raise ValueError("No data dictionary provided")

    datasets = []
    total_samples = 0
    
    for subj in subjects:
        if subj in data_dict:
            for file_path in data_dict[subj]:
                try:
                    dataset = NPZDataset(file_path, window_size_key)
                    datasets.append(dataset)
                    total_samples += len(dataset)
                    print(f"   Added {file_path.name}: {len(dataset)} samples")
                except Exception as e:
                    print(f"   ✗ Error loading {file_path.name}: {e}")
    
    if not datasets:
        raise ValueError(f"No valid data found for subjects: {subjects}")
    
    print(f"   Total: {len(datasets)} files, {total_samples} samples")
    return ConcatDataset(datasets)

# SUBJECT-LEVEL CROSS-VALIDATION

def create_patient_split(
    subject_ids: List[str], 
    n_train: int = N_TRAIN, 
    n_val: int = N_VAL, 
    n_test: int = N_TEST
) -> Tuple[List[str], List[str], List[str]]:
    """    
    Split subjects into train, validation, and test sets.
    
    Args:
        subject_ids: List of subject IDs to split
        n_train: Number of subjects for training
        n_val: Number of subjects for validation (0 = no validation)
        n_test: Number of subjects for testing (0 = no test set)
        
    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects)
        
    Raises:
        ValueError: If not enough subjects available for requested split
    """
    total_needed = n_train + n_val + n_test  
    n_subjects = len(subject_ids)

    if n_subjects < total_needed:
        raise ValueError(
            f"Not enough subjects for requested split!\n"
            f"  Available: {n_subjects} subjects\n"
            f"  Requested: {n_train} train + {n_val} val + {n_test} test = {total_needed}\n"
            f"  Please reduce N_TRAIN, N_VAL, or N_TEST in the script."
        )
    
    if n_train < 1:
        raise ValueError("N_TRAIN must be at least 1")

    shuffled = subject_ids.copy()
    np.random.shuffle(shuffled)
    
    # Split subjects into train, validation, and test sets
    train_end = n_train
    val_end = train_end + n_val
    test_end = val_end + n_test

    train_subj = shuffled[:train_end]
    val_subj = shuffled[train_end:val_end] if n_val > 0 else []
    test_subj = shuffled[val_end:test_end] if n_test > 0 else []
    
    return train_subj, val_subj, test_subj

# MODEL EVALUATION FUNCTION

def evaluate_model(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        Tuple of (average_loss, accuracy, f1_score)
    """
    # Check if loader is empty
    if len(loader) == 0:
        raise ValueError("DataLoader is empty")

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
    all_labels = torch.cat(all_labels, dim=0).numpy().flatten()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, acc, f1

def train_kan_model(KANSeizureDetector, train_loader, val_loader, epochs, lr, device, pos_weight):
    """
    Train the KAN model with optional validation.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader (can be None)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        pos_weight: Positive class weight for loss function
    """
    if len(train_loader) == 0:
        raise ValueError("No data in train loader")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            reg_loss = model.kan.regularization_loss(L1_LAMBDA, ENTROPY_LAMBDA)
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validate if validation loader exists
        if val_loader is not None and len(val_loader) > 0:
            val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        else:
            # No validation - just show training loss
            history['val_loss'].append(None)
            history['val_acc'].append(None)
            history['val_f1'].append(None)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
        
        scheduler.step()
        
    return history

if __name__ == "__main__":
    print("SUBJECT-LEVEL CROSS-VALIDATION")

    # Load data (file paths only, not actual data)
    print("\n[1/6] Loading data file paths...")
    data_by_subject = load_data(DATA_PATH, WINDOW_SIZE_KEY)
    
    if not data_by_subject:
        raise ValueError(f"No valid data found in {DATA_PATH}")
    
    subject_ids = list(data_by_subject.keys())
    total_files = sum(len(files) for files in data_by_subject.values())
    print(f"\n✓ Found {len(subject_ids)} subjects with {total_files} files total")
    print(f"  Subjects: {subject_ids}")

    # Split subjects into train, validation, and test sets
    print(f"\n[2/6] Splitting subjects (Train:{N_TRAIN}, Val:{N_VAL}, Test:{N_TEST})...")
    train_subj, val_subj, test_subj = create_patient_split(subject_ids)

    print(f"  Train subjects: {train_subj}")
    print(f"  Val subjects:   {val_subj if val_subj else 'None'}")
    print(f"  Test subjects:  {test_subj if test_subj else 'None'}")
    
    # Create datasets (lazy loading)
    print("\n[3/6] Creating datasets (lazy loading)...")
    print("  Training set:")
    train_dataset = create_dataset_from_subjects(train_subj, data_by_subject, WINDOW_SIZE_KEY)
    
    val_dataset = None
    if val_subj:
        print("  Validation set:")
        val_dataset = create_dataset_from_subjects(val_subj, data_by_subject, WINDOW_SIZE_KEY)
    else:
        print("  Validation set: None (N_VAL=0)")
    
    test_dataset = None
    if test_subj:
        print("  Test set:")
        test_dataset = create_dataset_from_subjects(test_subj, data_by_subject, WINDOW_SIZE_KEY)
    else:
        print("  Test set: None (N_TEST=0)")
    
    # Create dataloaders
    print("\n[4/6] Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  Train loader: {len(train_loader)} batches")
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"  Val loader: {len(val_loader)} batches")
    else:
        print(f"  Val loader: None")
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"  Test loader: {len(test_loader)} batches")
    else:
        print(f"  Test loader: None")

    # Set up model and weights
    print("\n Setting up model and weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_X, sample_Y = train_dataset[0]
    input_dim = sample_X.shape[0]
    print(f"  Input dimension: {input_dim}")
    
    # Calculate class imbalance from training data
    print("  Calculating class weights...")
    num_pos = 0
    total_samples = len(train_dataset)
    
    # Sample a subset if dataset is very large
    sample_size = min(10000, total_samples)
    if sample_size < total_samples:
        print(f"  Sampling {sample_size}/{total_samples} for class weight calculation...")
        indices = np.random.choice(total_samples, sample_size, replace=False)
    else:
        indices = range(total_samples)

    for i in indices:
        label = train_dataset[i][1].item()
        if label > 0:
            num_pos += 1

    num_neg = len(indices) - num_pos

    # Validate
    if num_pos == 0:
        raise ValueError("No positive samples in training data")
    if num_neg == 0:
        raise ValueError("No negative samples in training data")

    pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32)
    if pos_weight > 10.0:
        print(f"  WARNING: Very high pos_weight ({pos_weight:.2f}). Capping at 10.0")
        pos_weight = torch.clamp(pos_weight, max=10.0)
        
    print(f"  Class distribution: {num_pos} positive, {num_neg} negative")
    print(f"  Pos_weight: {pos_weight:.2f}")
    

    # Initialize Model
    print("\n  Initializing KAN model...")
    model = KANSeizureDetector(
        input_dim=input_dim, 
        hidden_layers=[64, 32], 
        grid_size=GRID_SIZE, 
        dropout=DROPOUT
    )
    model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run training
    print(f"\n[6/6] Training model ({NUM_EPOCHS} epochs)...")
    print("-" * 60)
    history = train_kan_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=NUM_EPOCHS, 
        lr=LEARNING_RATE, 
        device=device, 
        pos_weight=pos_weight
    )
    print("-" * 60)

    # Final test evaluation
    if test_loader is not None and len(test_loader) > 0:
        print("\n[Final] Evaluating on test set...")
        test_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, test_criterion, device)
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
    else:
        print("\n[Final] No test set available (N_TEST=0)")
        test_loss, test_acc, test_f1 = None, None, None


    # Save model and results
    print("\n[Saving] Saving model and results...")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'test_metrics': {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        },
        'subjects': {
            'train': train_subj,
            'val': val_subj if val_subj else [],
            'test': test_subj if test_subj else []
        },
        'hyperparameters': {
            'input_dim': input_dim,
            'hidden_layers': [64, 32],
            'grid_size': GRID_SIZE,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': NUM_EPOCHS,
            'window_size_key': WINDOW_SIZE_KEY
        }
    }
    
    model_path = project_root / 'kan_seizure_model.pth'
    torch.save(save_dict, model_path)
    print(f"  ✓ Model saved to '{model_path}'")


    # Plot Results
    plt.figure(figsize=(12, 6))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('KAN Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.plot(history['val_f1'], label='Val F1', color='red')
    plt.title('KAN Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = project_root / 'kan_training_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Plot saved to '{plot_path}'")
    plt.close()
    
    print("✓ Training complete!")
