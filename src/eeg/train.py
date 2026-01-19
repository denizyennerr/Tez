from os import path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from eeg.models import KANSeizureDetector


# HYPERPARAMETERS
SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
GRID_SIZE = 5
DROPOUT = 0.2
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATA_PATH = Path("data/preprocessed")
WINDOW_SIZE_KEY = '2s'
L1_LAMBDA = 1e-5
ENTROPY_LAMBDA = 1e-5

# Setting the seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# LOAD DATA FUNCTION
def load_data(data_path: Path, window_size_key: str) -> Dict[str, Dict[str, List]]:
    """
    Load data from the preprocessed directory.
    """
    data_by_subject = {}
    for file_path in data_path.glob("*.npz"):
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
                
        except (OSError, KeyError, ValueError, np.exceptions.DTypeError) as e:
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

    X_np = np.concatenate(X_list, axis=0)
    Y_np = np.concatenate(Y_list, axis=0)
    
    # Preprocessing
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32)
    
    # Flatten if necessary (Batch, Time, Channels) -> (Batch, Features)
    if X_tensor.dim() > 3:
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

# CREATE PATIENT SPLIT FUNCTION
def create_patient_split(subject_ids: List[str], train_ratio: float = 0.6, 
                        val_ratio: float = 0.2, test_ratio: float = 0.2):
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    n_test = n_subjects - n_train - n_val  # Ensure all subjects are used
    
    if n_subjects < n_train + n_val + n_test:
        raise ValueError(f"Not enough subjects: need {total_needed}, have {n_subjects}")
    
    shuffled = subject_ids.copy()
    np.random.shuffle(shuffled)
    
    train_subj = shuffled[:n_train]
    val_subj = shuffled[n_train:n_train + n_val]
    test_subj = shuffled[n_train + n_val:n_train + n_val + n_test]
    
    return train_subj, val_subj, test_subj

# # TRAINING LOOP FUNCTION

def train_kan_model(model, train_loader, val_loader, NUM_EPOCHS=10, lr=0.0001, device='cpu'):
    # Add history lists
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    print(f"Starting training on {device}...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # Add regularization loss (unique to KAN)
            reg_loss = model.kan.regularization_loss(L1_LAMBDA, ENTROPY_LAMBDA)
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            # Calculate accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_acc += (preds == y_batch).float().mean().item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

        # 2. Calculate averages
        if len(train_loader) == 0:
             raise ValueError("Train loader is empty")
        if len(val_loader) == 0:
            raise ValueError("Validation loader is empty")

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 3. Store them in the lists
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Metrics
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        scheduler.step()

    return model

# EVALUATE MODEL FUNCTION

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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, acc, f1


if __name__ == "__main__":
    print("SUBJECT-LEVEL CROSS-VALIDATION")

    # Load data
    print("\n Loading data...")
    data_by_subject = load_data(DATA_PATH, WINDOW_SIZE_KEY)
    
    subject_ids = list(data_by_subject.keys())
    print(f"   Found {len(subject_ids)} subjects: {subject_ids}")
    
    if len(subject_ids) < 5:
        raise ValueError(f"Need at least 5 subjects, but found only {len(subject_ids)}")
    
    # Split subjects: 3 train, 1 val, 1 test
    print("\n Splitting subjects (3 train, 1 val, 1 test)...")
    train_subj, val_subj, test_subj = create_patient_split(subject_ids, n_train=3, n_val=1, n_test=1)
    
    print(f"   Train subjects: {train_subj}")
    print(f"   Val subjects:   {val_subj}")
    print(f"   Test subjects:  {test_subj}")
    
    # Create datasets
    print("\ Creating datasets...")
    train_dataset = create_dataset_from_subjects(train_subj, data_by_subject)
    val_dataset = create_dataset_from_subjects(val_subj, data_by_subject)
    test_dataset = create_dataset_from_subjects(test_subj, data_by_subject)
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Create dataloaders
    print("\n4. Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get input dimension from first batch
    sample_X, sample_Y = train_dataset[0]
    input_dim = sample_X.shape[0]
    print(f"   Input dimension: {input_dim}")
    
    # Calculate class imbalance from training data
    all_train_labels = torch.cat([train_dataset[i][1] for i in range(len(train_dataset))])
    num_pos = int(all_train_labels.sum().item())
    num_neg = int(all_train_labels.size(0) - num_pos)
    pos_weight = torch.tensor(num_neg / max(num_pos, 1), dtype=torch.float32)
    
    
    print(f" Dataset imbalance: {num_pos} positive, {num_neg} negative")
    print(f" Pos_weight: {pos_weight:.2f}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Using device: {device}")

    # Initialize Model
    print("\n Initializing model...")
    model = KANSeizureDetector(input_dim=input_dim, hidden_layers=[64, 32], grid_size=GRID_SIZE, dropout=DROPOUT)
    model.to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # History lists
    print(f"\n. Starting training for {NUM_EPOCHS} epochs...")
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Add KAN regularization
            reg_loss = model.kan.regularization_loss(1e-5, 1e-5)
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()
        
        # Validation
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        
        # Calculate train loss
        train_loss = epoch_train_loss / len(train_loader)
        
        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_f1_history.append(val_f1)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}")
    
        scheduler.step()

    # Final test evaluation
    print("Final Test Evaluation")
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, device)
    print(f"\n   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test F1: {test_f1:.4f}")

    # Save model  
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': NUM_EPOCHS,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
        'val_f1_history': val_f1_history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'train_subjects': train_subj,
        'val_subjects': val_subj,
        'test_subjects': test_subj,
    }, 'kan_seizure_model.pth')
    print("   Model saved to 'kan_seizure_model.pth'")
    print("TRAINING COMPLETE!")
   

    # Plot Results
 
    plt.figure(figsize=(12, 6))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss', color='blue')
    plt.plot(val_loss_history, label='Val Loss', color='orange')
    plt.title('KAN Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Metrics
    plt.subplot(1, 2, 2)
    plt.plot(val_acc_history, label='Val Accuracy', color='green')
    plt.plot(val_f1_history, label='Val F1', color='red')
    plt.title('KAN Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kan_training_results.png')
    print("    Plot saved to 'kan_training_results.png'")
    plt.show()