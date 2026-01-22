import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# # Data Loading Functions
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent
src_path = project_root / 'src'
DATA_PATH = project_root / 'data' / 'preprocessed'

# # Add paths to sys.path
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))
# if str(src_path) not in sys.path:
#     sys.path.append(str(src_path))
# # Import model from the correct path
# try:
#     from src.eeg.models import KANSeizureDetector
#     print(" Success! Imported via 'src.eeg.models'")
# except ImportError:
#     try:
#         from eeg.models import KANSeizureDetector
#         print(" Success! Imported via 'eeg.models'")
#     except ImportError as e:
#         print(f" Failed to import model. Error: {e}")
#         sys.exit(1) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# KAN LINEAR MODEL

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Creating the spline grid
        h = (grid_range[1] - grid_range[0]) / grid_size # step size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1) # expand to the number of features
            .contiguous() # contiguous tensor
        )
        self.register_buffer("grid", grid) # save the grid in the model
        
        # Learnable parameters
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # base weight matrix
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order) # spline weight matrix
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features) # spline scaler
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters() 

    # Reset parameters

    def reset_parameters(self):
        # Initialize the base weight matrix
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Initialize the spline weights with small random noise
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                ) # Small random values centered at 0
                * self.scale_noise
                / self.grid_size
            ) # Convert noise to spline coefficients
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], # Use middle part of the grid to avoid boundary effects
                    noise, # Small random values centered at 0
                ) # Convert noise to spline coefficients
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline) # Initialize the spline scaler

    # B-splines function

    def b_splines(self, x: torch.Tensor):
        # Check if the input is valid
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Recursive Construction of B-splines
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    # Converting Curves to Coefficients

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    # Forward pass

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        # Base Transformation           
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # Spline Transformation
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    # Adaptive Grid Update

    @torch.no_grad() # no gradient computation
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x) # Current basis functions
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight # Current spline coefficients
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)
        # compute the current spline outputs before updating the grid

        # Sort the input values

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

        # Regularization Loss

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1) # L1 regularization on the spline weights
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation # Probability distribution
        regularization_loss_entropy = -torch.sum(p * p.log()) # Entropy regularization
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        ) # Regularization loss

        # KAN Multi-Layer network

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    # Forward pass

    def forward(self, x: torch.Tensor, update_grid=False):
        # Forward pass through each layer
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x) # Update the grid if needed
            x = layer(x) # Forward pass through the layer
        return x # Return the output

    # Aggregated Regularization Loss
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

# KAN Seizure Detector Wrapper

class KANSeizureDetector(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32], grid_size=5, dropout=0.2):
        super().__init__()
        # Input layer -> Hidden Layers -> 1 Output (Binary Classification)
        self.architecture = [input_dim] + hidden_layers + [1]
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        # KAN layers
        self.kan = KAN(
            layers_hidden=self.architecture,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )

    def forward(self, x, update_grid=False):
        # Flatten input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Apply dropout before KAN layers to prevent overfitting
        x = self.dropout(x)
        logits = self.kan(x, update_grid=update_grid)
        return logits


# NPZ Dataset Class
class NPZDataset(Dataset):
    """
    Lazy-loading Dataset for preprocessed NPZ files.
    Optimized to keep RAM usage low for CPU training.
    """
    def __init__(self, npz_path: Path, window_size_key: str = '2s'):
        self.npz_path = npz_path
        self.window_size_key = window_size_key
    
        # FIX: Use np.load with mmap_mode, not np.memmap directly
        try:
            self.mmap_file = np.load(npz_path, mmap_mode='r')
        except Exception as e:
            raise IOError(f"Failed to load {npz_path}: {e}")

        # FIX: Logic to resolve the correct key name
        if window_size_key in self.mmap_file:
            self.data_key = window_size_key
        elif 'data' in self.mmap_file:
            self.data_key = 'data'
        else:
            # Debug info: print available keys if lookup fails
            available_keys = list(self.mmap_file.keys())
            raise KeyError(f"Data key '{window_size_key}' not found in {npz_path.name}. Available keys: {available_keys}")
        
        if 'labels' not in self.mmap_file:
            raise KeyError(f"'labels' not found in {npz_path.name}")
        
        self.n_samples = len(self.mmap_file[self.data_key])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Access data using the resolved key
        data_sample = self.mmap_file[self.data_key][idx]
        label_sample = self.mmap_file['labels'][idx]
        
        data_tensor = torch.tensor(data_sample, dtype=torch.float32)
        label_tensor = torch.tensor(label_sample, dtype=torch.float32)
            
        # Flatten if necessary (Time, Channels) -> (Features)
        if data_tensor.dim() >= 2:
            data_tensor = data_tensor.flatten()
        
        # Ensure label is 1D with single element
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)
        elif label_tensor.numel() > 1:
            label_tensor = label_tensor.flatten()[0:1]
        
        # FIX: Return tuple (data, label), not the file object
        return data_tensor, label_tensor

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
NUM_WORKERS = 4 if os.cpu_count() > 4 else 0

# SUBJECT-LEVEL CROSS-VALIDATION SPLIT
N_TRAIN = 3 
N_VAL = 1
N_TEST = 1

# Setting the seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# LOAD DATA FUNCTION - Returns file paths organized by subject
def load_data(data_path: Path) -> Dict[str, List[Path]]:
    """
    Load file paths from the preprocessed directory, organized by subject.
    Does NOT load actual data into memory - returns paths for lazy loading.
    """
    data_by_subject = {}
    
    for file_path in data_path.glob("**/*.npz"):
        subject_id = file_path.stem.split('_')[0]
        if subject_id not in data_by_subject:
            data_by_subject[subject_id] = []
        data_by_subject[subject_id].append(file_path)
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
    if not subjects: return None

    datasets = []
    total_samples = 0
    
    for subj in subjects:
        if subj in data_dict:
            for file_path in data_dict[subj]:
                try:
                    dataset = NPZDataset(file_path, window_size_key)
                    datasets.append(dataset)
                    total_samples += len(dataset)
                    print(f" Added {file_path.name}: {len(dataset)} samples")
                except Exception as e:
                    print(f" Error loading {file_path.name}: {e}")
    
    if not datasets:
        raise ValueError(f"No valid data found for subjects: {subjects}")
    
    print(f"   Total: {len(datasets)} files, {total_samples} samples")
    return ConcatDataset(datasets)

# SUBJECT-LEVEL CROSS-VALIDATION

def create_patient_split(
    subject_ids: List[str], 
    total_needed: int = N_TRAIN + N_VAL + N_TEST
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

    if len(subject_ids) < total_needed:
        raise ValueError(
            f"Not enough subjects for requested split!\n"
        )
    
    shuffled = subject_ids.copy()
    np.random.shuffle(shuffled) 
    
    # Split subjects into train, validation, and test sets
    train_end = N_TRAIN 
    val_end = train_end + N_VAL
    test_end = val_end + N_TEST

    train_subj = shuffled[:train_end] 
    val_subj = shuffled[train_end:val_end] if N_VAL > 0 else [] 
    test_subj = shuffled[val_end:test_end] if N_TEST > 0 else [] 
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
    # data_by_subject = load_data(DATA_PATH)
    data_by_subject = load_data(Path('data/preprocessed'))
    subject_ids = list(data_by_subject.keys())
    train_subj, val_subj, test_subj = create_patient_split(subject_ids)

    # Create datasets from subjects
    train_dataset = create_dataset_from_subjects(train_subj, data_by_subject, WINDOW_SIZE_KEY)
    val_dataset = create_dataset_from_subjects(val_subj, data_by_subject, WINDOW_SIZE_KEY)
    test_dataset = create_dataset_from_subjects(test_subj, data_by_subject, WINDOW_SIZE_KEY)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0)) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0)) if test_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    # Calculate class imbalance from training data
    print("\n Estimating class weights...")
    
    # Collect all labels
    all_labels = []
    for X_batch, y_batch in train_loader:
        all_labels.append(y_batch.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)

    num_pos = (all_labels == 1).sum()
    num_neg = (all_labels == 0).sum()

    if num_pos == 0:
        print(" No positive samples found in subset. Using default pos_weight=1.0")
        pos_weight = torch.tensor(1.0, dtype=torch.float32)
    else:
        pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32)
        if pos_weight > 10.0:
            print(f"Very high pos_weight ({pos_weight:.2f}). Capping at 10.0")
            pos_weight = torch.clamp(pos_weight, max=10.0)

    print(f"  Class distribution: {num_pos} positive, {num_neg} negative")
    print(f"  Pos_weight: {pos_weight:.2f}")
    
    sample_X, sample_Y = train_dataset[0]
    input_dim = sample_X.shape[0]
    print(f"  Input dimension: {input_dim}")

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
    print(f"\n Training model ({NUM_EPOCHS} epochs)...")
    history = train_kan_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=NUM_EPOCHS, 
        lr=LEARNING_RATE, 
        device=device, 
        pos_weight=pos_weight
    )

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
    print("\n Saving model and results...")
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
            'epochs': NUM_EPOCHS,
            'window_size_key': WINDOW_SIZE_KEY
        }
    }
    
    model_path = Path('kan_seizure_model.pth')
    torch.save(save_dict, model_path)
    print(f" Model saved to '{model_path}'")


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
    plot_path = Path('kan_training_results.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Plot saved to '{plot_path}'")
    plt.close()
    
    print("✓ Training complete!")
