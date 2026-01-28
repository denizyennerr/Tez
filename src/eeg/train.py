import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# Configuration
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
DATA_PATH = project_root / 'data' / 'preprocessed'


# Utilize the preprocessed data
class EEGDataset(Dataset):
    def __init__(self, file_list, window_key='2s'):
        self.index_map = []
        for f_path in file_list:
            with np.load(f_path) as f:
                key = window_key if window_key in f else 'data'
                labels = f['labels']
                for i in range(len(labels)):
                    if labels[i] in (0, 1):
                        self.index_map.append({
                            'file': str(f_path),
                            'key': key,
                            'idx': i,
                            'label': float(labels[i])
                        })
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        info = self.index_map[idx]
        with np.load(info['file']) as f:
            data = f[info['key']][info['idx']]
        if data.ndim > 1:
            data = data.flatten()
        # Normalize
        m, s = data.mean(), data.std()
        if s > 1e-8:
            data = (data - m) / s
        return torch.from_numpy(data.astype(np.float32)), torch.tensor([info['label']], dtype=torch.float32)

def undersample_indices(index_map, ratio=1.0):
    """Undersample majority class to achieve target ratio."""
    seizure_idx = [i for i, x in enumerate(index_map) if x['label'] == 1]
    non_seizure_idx = [i for i, x in enumerate(index_map) if x['label'] == 0]
    
    # Keep all seizures, undersample non-seizures
    target_non_seizure = int(len(seizure_idx) * ratio)
    sampled_non_seizure = random.sample(non_seizure_idx, min(target_non_seizure, len(non_seizure_idx)))
    
    return seizure_idx + sampled_non_seizure

# Simple model matching KAN structure
class KANSeizureModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32]):
        super().__init__()
        layers = [input_dim] + hidden_layers + [1]
        self.kan = KANLayer(layers)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten
        return self.kan(x)

# Training and validation functions
def train_and_validate_model(model, epochs, learning_rate, train_loader, val_loader, model_name):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pos_weight = torch.tensor([5.0])
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            predicted_y = model(x)
            loss = loss_fn(predicted_y, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                predicted_y = model(x)
                val_loss = loss_fn(predicted_y, y.unsqueeze(1))
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch}, {model_name} Train Loss: {avg_loss}, Validation Loss: {avg_val_loss}")

# Evaluation function
def evaluate_model(model, eval_loader, model_name):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in eval_loader:
            predicted_y = model(x)
            predictions.extend(predicted_y.squeeze().cpu().numpy())
            actuals.extend(y.cpu().numpy())
    return predictions, actuals

# Prepare dataset and loaders
all_files = list(DATA_PATH.glob("**/*.npz"))
dataset = EEGDataset(all_files)

# First split by patient, then undersample training set
balanced_indices = undersample_indices(dataset.index_map, ratio=5.0)
balanced_dataset = Subset(dataset, balanced_indices)

# Split the balanced dataset
train_size = int(0.7 * len(balanced_dataset))
val_size = len(balanced_dataset) - train_size
print(f"Train size: {train_size}, Val size: {val_size}")

train_dataset, val_dataset = torch.utils.data.random_split(balanced_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model layers
sample_x, _ = dataset[0]
input_dim = sample_x.shape[0]  # Get actual feature dimension
model = KANSeizureModel(input_dim=input_dim, hidden_layers=[64, 32])

# Initialize and train the KAN model
train_and_validate_model(model, epochs=50, learning_rate=0.001, train_loader=train_loader, val_loader=val_loader, model_name="KAN")

# Evaluate both models
kan_predictions, kan_actuals = evaluate_model(model, val_loader, "KAN")

# Log results
kan_data = [[pred, act] for pred, act in zip(kan_predictions, kan_actuals)]

# Visualize the results
plt.figure(figsize=(10, 5))
plt.plot(kan_predictions, label='Predictions')
plt.plot(kan_actuals, label='Actuals')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('KAN Model Predictions vs Actuals')
plt.legend()
# Save model states
plt.tight_layout()
output_path = project_root / 'training_results1.png'
plt.savefig(output_path, dpi=150)
plt.close()
print(f"\nPlot saved to: {output_path}")
    

