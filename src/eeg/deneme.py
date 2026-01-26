"""
EEG Seizure Detection Training Script
Uses KANSeizureDetector model with PyTorch DataLoader
Optimized for 16GB RAM with batch_size 16
"""
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Import model
try:
    from eeg.models import KANSeizureDetector
except ImportError:
    current_file = Path(__file__).resolve()
    src_path = current_file.parent.parent.parent / 'src'
    sys.path.append(str(src_path))
    from eeg.models import KANSeizureDetector

# Data path configuration
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
DATA_PATH = project_root / 'data' / 'preprocessed'


class EEGDataset(Dataset):
    """
    Memory-efficient EEG dataset with file caching.
    
    Strategy for 16GB RAM:
    - Build index map once during init (file paths + sample indices)
    - Cache loaded file data to avoid repeated file I/O
    - Use LRU-style cache with max_cached_files limit
    """
    
    def __init__(self, file_list: list, window_size_key: str = '2s', max_cached_files: int = 5):
        """
        Args:
            file_list: List of Path objects to NPZ files
            window_size_key: Key for windowed data in NPZ files
            max_cached_files: Max number of NPZ files to keep in memory cache
        """
        self.file_list = file_list
        self.window_size_key = window_size_key
        self.max_cached_files = max_cached_files
        self.index_map = []
        
        # File cache: {file_path_str: {'data': array, 'labels': array}}
        self._cache = {}
        self._cache_order = []  # Track access order for LRU eviction
        
        # Build index map (file path + sample index)
        print(f"  Building index map from {len(file_list)} files...")
        for f_path in file_list:
            try:
                with np.load(f_path, allow_pickle=True) as f:
                    key = self.window_size_key if self.window_size_key in f else 'data'
                    if key not in f or 'labels' not in f:
                        continue
                    
                    n_samples = f[key].shape[0]
                    labels = f['labels']
                    
                    for i in range(n_samples):
                        if labels[i] in (0, 1):  # Valid binary labels only
                            self.index_map.append({
                                'file_path': str(f_path),
                                'data_key': key,
                                'index': i,
                                'label': float(labels[i])
                            })
            except Exception as e:
                print(f"  Warning: Could not load {f_path.name}: {e}")
                continue
        
        print(f"  Index map built: {len(self.index_map)} samples from {len(file_list)} files")

    def __len__(self):
        return len(self.index_map)
    
    def _load_file_to_cache(self, file_path: str, data_key: str):
        """Load file data into cache with LRU eviction."""
        if file_path in self._cache:
            # Move to end of access order (most recently used)
            if file_path in self._cache_order:
                self._cache_order.remove(file_path)
            self._cache_order.append(file_path)
            return
        
        # Evict oldest if cache is full
        while len(self._cache) >= self.max_cached_files and self._cache_order:
            oldest = self._cache_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
            gc.collect()
        
        # Load new file
        with np.load(file_path, allow_pickle=True) as f:
            self._cache[file_path] = {
                'data': f[data_key][:].astype(np.float32),
                'labels': f['labels'][:].astype(np.float32)
            }
        self._cache_order.append(file_path)
    
    def __getitem__(self, idx):
        """Get sample with file caching for efficiency."""
        if idx < 0 or idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range [0, {len(self.index_map)})")
        
        info = self.index_map[idx]
        file_path = info['file_path']
        data_key = info['data_key']
        sample_idx = info['index']
        
        # Ensure file is in cache
        self._load_file_to_cache(file_path, data_key)
        
        # Get data from cache
        data = self._cache[file_path]['data'][sample_idx]
        label = info['label']
        
        # Flatten if multi-dimensional
        if data.ndim > 1:
            data = data.flatten()
        
        # Z-score normalization per sample
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-8:
            data = (data - mean) / std
        
        return (
            torch.from_numpy(data.copy()),
            torch.tensor([label], dtype=torch.float32)
        )
    
    def clear_cache(self):
        """Explicitly clear the file cache."""
        self._cache.clear()
        self._cache_order.clear()
        gc.collect()
    
    @staticmethod
    def get_patient_id(file_path: Path) -> str:
        """Extract patient ID from filename."""
        name = file_path.name
        if 'chb' in name.lower():
            # Format: chb01_03.npz -> chb01
            parts = name.split('_')
            return parts[0] if parts else name
        return name.split('_')[0] if '_' in name else name


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("KAN SEIZURE DETECTION - Training Script")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # HYPERPARAMETERS
    # -------------------------------------------------------------------------
    BATCH_SIZE = 16
    NUM_WORKERS = 0  # Windows compatibility
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    WINDOW_SIZE_KEY = '2s'
    MAX_BATCHES_PER_EPOCH = 16  # Limit batches per epoch for quick iteration
    MAX_CACHED_FILES = 5  # Number of NPZ files to keep in memory
    
    # -------------------------------------------------------------------------
    # DEVICE SETUP
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # -------------------------------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------------------------------
    print(f"\nData path: {DATA_PATH}")
    all_files = list(DATA_PATH.glob("**/*.npz"))
    print(f"Found {len(all_files)} NPZ files")
    
    if not all_files:
        print("ERROR: No NPZ files found!")
        sys.exit(1)
    
    # Extract patient IDs for group-based splitting
    groups = [EEGDataset.get_patient_id(f) for f in all_files]
    unique_patients = list(set(g for g in groups if g is not None))
    print(f"Found {len(unique_patients)} unique patients: {unique_patients[:5]}...")
    
    # -------------------------------------------------------------------------
    # TRAIN/VALIDATION SPLIT (Patient-level, no data leakage)
    # -------------------------------------------------------------------------
    if len(unique_patients) < 2:
        print("WARNING: Not enough patients for proper split, using all data for both")
        train_files = all_files
        val_files = all_files
    else:
        # Use GroupKFold for patient-level split
        cv = GroupKFold(n_splits=min(5, len(unique_patients)))
        train_idx, val_idx = next(cv.split(all_files, groups=groups))
        
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        
        # Verify no patient leakage
        train_patients = set(EEGDataset.get_patient_id(f) for f in train_files)
        val_patients = set(EEGDataset.get_patient_id(f) for f in val_files)
        overlap = train_patients & val_patients
        if overlap:
            print(f"WARNING: Patient overlap detected: {overlap}")
        else:
            print("Patient-level split verified (no leakage)")
    
    print(f"\nTrain files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    
    # -------------------------------------------------------------------------
    # CREATE DATASETS
    # -------------------------------------------------------------------------
    print("\nCreating datasets...")
    train_dataset = EEGDataset(train_files, WINDOW_SIZE_KEY, max_cached_files=MAX_CACHED_FILES)
    val_dataset = EEGDataset(val_files, WINDOW_SIZE_KEY, max_cached_files=MAX_CACHED_FILES)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # CREATE DATALOADERS
    # -------------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # CPU training
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    print(f"\nTrain batches: {len(train_loader)} (using up to {MAX_BATCHES_PER_EPOCH}/epoch)")
    print(f"Val batches: {len(val_loader)}")
    
    # -------------------------------------------------------------------------
    # MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    sample_x, sample_y = train_dataset[0]
    input_dim = sample_x.shape[0]
    print(f"\nInput dimension: {input_dim}")
    
    model = KANSeizureDetector(
        input_dim=input_dim,
        hidden_layers=[64, 32],
        grid_size=5,
        dropout=0.2
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # -------------------------------------------------------------------------
    # LOSS AND OPTIMIZER
    # -------------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training: {NUM_EPOCHS} epochs, {MAX_BATCHES_PER_EPOCH} batches/epoch")
    print(f"{'='*60}")
    
    # Tracking lists
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(NUM_EPOCHS):
        # ----- TRAINING PHASE -----
        model.train()
        batch_losses = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if batch_idx >= MAX_BATCHES_PER_EPOCH:
                break
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
            # Memory cleanup
            del X_batch, y_batch, y_pred, loss
        
        avg_train_loss = np.mean(batch_losses) if batch_losses else 0.0
        history['train_loss'].append(avg_train_loss)
        
        # ----- VALIDATION PHASE -----
        model.eval()
        val_batch_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if batch_idx >= MAX_BATCHES_PER_EPOCH:
                    break
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_batch_losses.append(loss.item())
                
                preds = (torch.sigmoid(y_pred) > 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
                
                del X_batch, y_batch, y_pred, loss, preds
        
        avg_val_loss = np.mean(val_batch_losses) if val_batch_losses else 0.0
        val_acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
        val_f1 = f1_score(all_targets, all_preds, zero_division=0) if all_targets else 0.0
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        scheduler.step()
        gc.collect()
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f}")
    
    # -------------------------------------------------------------------------
    # FINAL RESULTS
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Final Val F1: {history['val_f1'][-1]:.4f}")
    
    if history['val_acc']:
        best_epoch = np.argmax(history['val_acc']) + 1
        print(f"Best Val Accuracy: {max(history['val_acc']):.4f} (Epoch {best_epoch})")
    
    # -------------------------------------------------------------------------
    # SAVE RESULTS PLOT
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    plt.plot(history['val_loss'], 'r-o', label='Val Loss', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KAN Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], 'g-o', label='Val Accuracy', markersize=4)
    plt.plot(history['val_f1'], 'm-o', label='Val F1', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('KAN Validation Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = project_root / 'training_results1.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {output_path}")
    
    # Cleanup
    train_dataset.clear_cache()
    val_dataset.clear_cache()
    gc.collect()
    print("\nDone!")
