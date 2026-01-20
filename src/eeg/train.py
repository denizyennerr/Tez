import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

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
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
GRID_SIZE = 5
DROPOUT = 0.2
WINDOW_SIZE_KEY = '2s'
L1_LAMBDA = 1e-5
ENTROPY_LAMBDA = 1e-5

# SUBJECT-LEVEL CROSS-VALIDATION
N_TRAIN = 3
N_VAL = 1
N_TEST = 1

# Setting the seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# LOAD DATA FUNCTION
def load_data(data_path: Path, window_size_key: str) -> Dict[str, Dict[str, List]]:
    """
    Load data from the preprocessed directory.
    """
    data_by_subject = {}
    for file_path in data_path.glob("**/*.npz"):
        try:
            # Extract subject ID 
            subject_id = file_path.stem.split('_')[0]
            
            if subject_id not in data_by_subject:
                data_by_subject[subject_id] = {'X': [], 'Y': []}
            
            data = np.load(file_path)
            
            if window_size_key in data and 'labels' in data:
                data_by_subject[subject_id]['X'].append(data[window_size_key])
                data_by_subject[subject_id]['Y'].append(data['labels'])
                print(f" Loaded {file_path.name} for subject {subject_id}")
            else:
                print(f" Skipping {file_path.name}: Missing '{window_size_key}' or 'labels'")
                
        except (OSError, KeyError) as e:
            print(f" Error loading {file_path.name}: {e}")
    
    return data_by_subject

# Create datasets from subject splits
def create_dataset_from_subjects(subjects: List[str], data_dict: Dict) -> TensorDataset:
    """Concatenates data for specific subjects and returns a TensorDataset."""
    if not subjects:
        raise ValueError("No subjects provided")
    if not data_dict:
        raise ValueError("No data provided")

    X_list, Y_list = [], []
    
    for subj in subjects:
        if subj in data_dict:
            X_list.extend(data_dict[subj]['X'])
            Y_list.extend(data_dict[subj]['Y'])
    
    if not X_list or not Y_list:
        raise ValueError("No data found for any subject")

    X_np = np.concatenate(X_list, axis=0)
    Y_np = np.concatenate(Y_list, axis=0)
    
    # Preprocessing
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32)
    
    # Flatten if necessary (Batch, Time, Channels) -> (Batch, Features)
    if X_tensor.dim() == 3:
        X_tensor = X_tensor.view(X_tensor.size(0), -1)
    elif X_tensor.dim() == 2:
        pass
    else:
        raise ValueError(f"Invalid dimensions: {X_tensor.dim()}")
        
    # Ensure labels are binary and correct shape
    Y_tensor = (Y_tensor > 0).float()
    if Y_tensor.dim() == 1:
        Y_tensor = Y_tensor.view(-1, 1)
        
    return TensorDataset(X_tensor, Y_tensor)

# SUBJECT-LEVEL CROSS-VALIDATION

def create_patient_split(subject_ids: List[str], n_train: int = N_TRAIN, n_val: int = N_VAL, n_test: int = N_TEST) -> Tuple[List[str], List[str], List[str]]:
    """    
    Split subjects into train, validation, and test sets.
    """
    total_needed = n_train + n_val + n_test  
    n_subjects = len(subject_ids)

    if n_subjects < total_needed:
        raise ValueError(f"Not enough subjects: need {total_needed}, have {n_subjects}")

    shuffled = subject_ids.copy()
    np.random.shuffle(shuffled)
    
    # Split subjects into train, validation, and test sets
    train_end = n_train
    val_end = train_end + n_val
    test_end = n_train + n_val + n_test

    train_subj = shuffled[:train_end]
    val_subj = shuffled[train_end:val_end]
    test_subj = shuffled[val_end:test_end]
    
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

def train_kan_model(model, train_loader, val_loader, epochs, lr, device, pos_weight):
    if len(train_loader) == 0:
        raise ValueError("No data in train loader")
    if len(val_loader) == 0:
        raise ValueError("No data in val loader")
    
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
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        scheduler.step()
        
    return history

if __name__ == "__main__":
    print("SUBJECT-LEVEL CROSS-VALIDATION")

    # Load data
    print("\n Loading data...")
    data_by_subject = load_data(DATA_PATH, WINDOW_SIZE_KEY)
    
    subject_ids = list(data_by_subject.keys())
    print(f"   Found {len(subject_ids)} subjects: {subject_ids}")

    # Split subjects into train, validation, and test sets
    train_subj, val_subj, test_subj = create_patient_split(subject_ids)

    print(f" Train subjects: {train_subj}")
    print(f" Val subjects:   {val_subj}")
    print(f" Test subjects:  {test_subj}")
    
    # Create datasets
    print("\n Creating datasets...")
    train_dataset = create_dataset_from_subjects(train_subj, data_by_subject)
    val_dataset = create_dataset_from_subjects(val_subj, data_by_subject)
    test_dataset = create_dataset_from_subjects(test_subj, data_by_subject)
    
    # Create dataloaders
    print("\n Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Set up model and weights
    print("\n Setting up model and weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_X, sample_Y = train_dataset[0]
    input_dim = sample_X.shape[0]
    
    
    # Calculate class imbalance from training data
    num_pos = 0
    total_samples = len(train_dataset)

    for i in range(total_samples):
        label = train_dataset[i][1].item()
        if label > 0:
            num_pos += 1

    num_neg = total_samples - num_pos

    # Validate
    if num_pos == 0:
        raise ValueError("No positive samples in training data")
    if num_neg == 0:
        raise ValueError("No negative samples in training data")

    pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32)
    if pos_weight > 10.0:
        print(f"   WARNING: Very high pos_weight ({pos_weight:.2f}). Capping at 10.0")
    pos_weight = torch.clamp(pos_weight, max=10.0)  
        
    print(f" Dataset imbalance: {num_pos} positive, {num_neg} negative")
    print(f" Pos_weight: {pos_weight:.2f}")
    

    # Initialize Model
    print("\n Initializing model...")
    model = KANSeizureDetector(input_dim=input_dim, hidden_layers=[64, 32], grid_size=GRID_SIZE, dropout=DROPOUT)
    model.to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run training
    history = train_kan_model(model, train_loader, val_loader, epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device, pos_weight=pos_weight)

    # Final test evaluation
    test_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, test_criterion, device)
    print(f"\n   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test F1: {test_f1:.4f}")


    # Save model and results
    print("\n Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'test_metrics': {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        },
        'subjects': {
            'train': train_subj,
            'val': val_subj,
            'test': test_subj
        },
        'hyperparameters': {
            'input_dim': input_dim,
            'hidden_layers': [64, 32],
            'grid_size': GRID_SIZE,
            'dropout': DROPOUT
        }
    }, 'kan_seizure_model.pth')

    print("   Model saved to 'kan_seizure_model.pth'")


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
    plt.savefig('kan_training_results.png')
    print("    Plot saved to 'kan_training_results.png'")
    plt.show()