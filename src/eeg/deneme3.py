"""
EEG Seizure Detection - CPU OPTIMIZED (CORRECTED)
Features:
1. Subject-Level Cross-Validation (GroupKFold)
2. RAM Efficiency: Contiguous batch sampling + LRU cache
3. Configurable batch limit for quick experiments
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
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import Model
try:
    from eeg.models import KANSeizureDetector
except ImportError:
    current_file = Path(__file__).resolve()
    src_path = current_file.parent.parent.parent / 'src'
    sys.path.append(str(src_path))
    from eeg.models import KANSeizureDetector

# Configuration
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
DATA_PATH = project_root / 'data' / 'preprocessed'


class ContiguousBatchSampler:
    """
    Yields batches of indices file-by-file to minimize disk seeking.
    Essential for training on HDD/CPU to prevent cache thrashing.
    """
    def __init__(self, dataset, batch_size, max_batches=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.shuffle = shuffle
        
        # Group indices by file path
        self.file_to_indices = defaultdict(list)
        for i, info in enumerate(dataset.index_map):
            self.file_to_indices[info['file_path']].append(i)
        self.file_paths = list(self.file_to_indices.keys())

    def __iter__(self):
        # Shuffle file order
        if self.shuffle:
            random.shuffle(self.file_paths)
        
        batch = []
        batches_yielded = 0
        
        for file_path in self.file_paths:
            indices = self.file_to_indices[file_path]
            
            # Shuffle indices within file
            if self.shuffle:
                random.shuffle(indices)
            
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    batches_yielded += 1
                    
                    # Stop if max_batches reached
                    if self.max_batches and batches_yielded >= self.max_batches:
                        return
        
        # Yield remaining samples
        if batch and (not self.max_batches or batches_yielded < self.max_batches):
            yield batch
    
    def __len__(self):
        if self.max_batches:
            return self.max_batches
        return len(self.dataset) // self.batch_size


# Dataset with LRU cache
class EEGDataset(Dataset):
    """
    Memory-efficient EEG dataset with LRU file caching.
    """
    def __init__(self, file_list, window_key='2s', max_cached_files=20):
        self.window_key = window_key
        self.max_cached_files = max_cached_files
        self.index_map = []
        self._cache = OrderedDict()
        
        print(f"    Indexing {len(file_list)} files...")
        for f_path in file_list:
            try:
                with np.load(f_path, mmap_mode='r') as f:
                    key = window_key if window_key in f else 'data'
                    if key not in f or 'labels' not in f:
                        continue
                    
                    labels = f['labels']
                    # Vectorized search for valid labels
                    valid_indices = np.where((labels == 0) | (labels == 1))[0]
                    
                    for i in valid_indices:
                        self.index_map.append({
                            'file_path': str(f_path),
                            'data_key': key,
                            'index': int(i),
                            'label': float(labels[i])
                        })
            except Exception as e:
                print(f"      Warning: Skipping {f_path.name}: {e}")
                continue
        
        print(f"      Total samples: {len(self.index_map)}")
    
    def __len__(self):
        return len(self.index_map)
    
    def _load_to_cache(self, file_path, data_key):
        """Load file to cache with LRU eviction"""
        if file_path in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(file_path)
            return
        
        # Evict oldest if cache is full
        while len(self._cache) >= self.max_cached_files:
            self._cache.popitem(last=False)
        
        # Load file
        with np.load(file_path, allow_pickle=True) as f:
            self._cache[file_path] = f[data_key][:].astype(np.float32)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range")
        
        info = self.index_map[idx]
        
        try:
            # Load to cache if needed
            self._load_to_cache(info['file_path'], info['data_key'])
            
            # Get from cache
            data = self._cache[info['file_path']][info['index']]
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1), torch.tensor([0.0])
        
        # Flatten if multidimensional
        if data.ndim > 1:
            data = data.flatten()
        
        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Z-score normalization
        mean, std = data.mean(), data.std()
        if std > 1e-8:
            data = (data - mean) / std
        
        return (
            torch.from_numpy(data.copy()),
            torch.tensor([info['label']], dtype=torch.float32)
        )
    
    def clear_cache(self):
        """Explicitly clear file cache"""
        self._cache.clear()
        gc.collect()
    
    @staticmethod
    def get_patient_id(file_path: Path) -> str:
        """Extract patient ID from filename"""
        name = file_path.name
        if 'chb' in name.lower():
            return name.split('_')[0]
        return name.split('_')[0] if '_' in name else name


# Main execution
if __name__ == '__main__':
    print("=" * 70)
    print("KAN Seizure Detection - CPU Optimized CV")
    print("=" * 70)
    
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 3
    MAX_BATCHES_PER_EPOCH = 16  # Limit for quick experiments
    MAX_CACHED_FILES = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load files
    print(f"\nData path: {DATA_PATH}")
    all_files = list(DATA_PATH.glob("**/*.npz"))
    
    if not all_files:
        print("ERROR: No NPZ files found!")
        sys.exit(1)
    
    print(f"Found {len(all_files)} NPZ files")
    
    # Extract patient groups
    groups = [EEGDataset.get_patient_id(f) for f in all_files]
    unique_patients = list(set(g for g in groups if g is not None))
    print(f"Found {len(unique_patients)} unique patients")
    
    if len(unique_patients) < 2:
        print("ERROR: Need at least 2 patients for cross-validation")
        sys.exit(1)
    
    # Cross-validation setup
    n_splits = min(5, len(unique_patients))
    cv = GroupKFold(n_splits=n_splits)
    
    print(f"\n{'='*70}")
    print(f"RUNNING {n_splits}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    all_fold_results = []
    
    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(all_files, groups=groups)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*70}")
        
        # Split files
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        
        # Verify no patient leakage
        train_patients = set(EEGDataset.get_patient_id(f) for f in train_files)
        val_patients = set(EEGDataset.get_patient_id(f) for f in val_files)
        overlap = train_patients & val_patients
        
        if overlap:
            print(f"  WARNING: Patient overlap detected: {overlap}")
        else:
            print(f"  ✓ No patient leakage")
        
        print(f"  Train files: {len(train_files)}")
        print(f"  Val files: {len(val_files)}")
        
        # Create datasets
        print(f"\n  Creating datasets...")
        train_dataset = EEGDataset(train_files, window_key='2s', max_cached_files=MAX_CACHED_FILES)
        val_dataset = EEGDataset(val_files, window_key='2s', max_cached_files=MAX_CACHED_FILES)
        
        if len(train_dataset) == 0:
            print(f"  ERROR: No training samples in fold {fold_idx + 1}!")
            continue
        
        # Calculate class weights from index map (fast)
        labels = [info['label'] for info in train_dataset.index_map]
        num_neg = labels.count(0.0)
        num_pos = labels.count(1.0)
        pos_weight = (num_neg / max(num_pos, 1)) * 2.0  # 2x boost for minority class
        
        print(f"    Class distribution: {num_neg} non-seizure, {num_pos} seizure")
        print(f"    Pos_weight: {pos_weight:.2f}")
        
        # Create samplers
        train_sampler = ContiguousBatchSampler(
            train_dataset, BATCH_SIZE, 
            max_batches=MAX_BATCHES_PER_EPOCH, 
            shuffle=True
        )
        val_sampler = ContiguousBatchSampler(
            val_dataset, BATCH_SIZE,
            max_batches=MAX_BATCHES_PER_EPOCH,
            shuffle=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=0
        )
        
        print(f"    Train batches: {len(train_sampler)} (limited to {MAX_BATCHES_PER_EPOCH})")
        print(f"    Val batches: {len(val_sampler)} (limited to {MAX_BATCHES_PER_EPOCH})")
        
        # Initialize model for this fold
        sample_x, sample_y = train_dataset[0]
        input_dim = sample_x.shape[0]
        
        print(f"\n  Initializing model (input_dim={input_dim})...")
        model = KANSeizureDetector(
            input_dim=input_dim,
            hidden_layers=[32, 16],
            grid_size=3,
            dropout=0.3
        ).to(device)
        
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
        )
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        
        # Training loop
        print(f"\n  {'='*68}")
        print(f"  TRAINING FOLD {fold_idx + 1}")
        print(f"  {'='*68}\n")
        
        fold_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': []
        }
        
        best_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            # Training phase
            model.train()
            batch_losses = []
            
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
            
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
                
                if batch_idx % 10 == 0:
                    pbar.set_postfix({'loss': f'{np.mean(batch_losses[-10:]):.4f}'})
                
                # Memory cleanup
                if batch_idx % 10 == 0:
                    del x, y, output, loss
                    gc.collect()
            
            avg_train_loss = np.mean(batch_losses) if batch_losses else 0.0
            fold_history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_batch_losses = []
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    
                    output = model(x)
                    loss = criterion(output, y)
                    val_batch_losses.append(loss.item())
                    
                    preds = (torch.sigmoid(output) > 0.5).float()
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_targets.extend(y.cpu().numpy().flatten())
                    
                    del x, y, output, loss, preds
            
            avg_val_loss = np.mean(val_batch_losses) if val_batch_losses else 0.0
            val_acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
            val_f1 = f1_score(all_targets, all_preds, zero_division=0) if all_targets else 0.0
            val_precision = precision_score(all_targets, all_preds, zero_division=0) if all_targets else 0.0
            val_recall = recall_score(all_targets, all_preds, zero_division=0) if all_targets else 0.0
            
            fold_history['val_loss'].append(avg_val_loss)
            fold_history['val_acc'].append(val_acc)
            fold_history['val_f1'].append(val_f1)
            fold_history['val_precision'].append(val_precision)
            fold_history['val_recall'].append(val_recall)
            
            scheduler.step()

            # Print epoch results
            print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"F1: {val_f1:.4f} | Acc: {val_acc:.4f}")
            gc.collect()
            
            # Early stopping
            # if val_f1 > best_f1:
            #     best_f1 = val_f1
            #     patience_counter = 0
            #     torch.save(model.state_dict(), project_root / f'best_model_fold{fold_idx+1}.pth')
            #     improvement_flag = "✓ NEW BEST"
            # else:
            #     patience_counter += 1
            #     improvement_flag = f"(no improvement: {patience_counter}/{PATIENCE})"
            
            # print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
            #       f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
            #       f"F1: {val_f1:.4f} | Acc: {val_acc:.4f} | {improvement_flag}")
            
            # if patience_counter >= PATIENCE:
            #     print(f"  Early stopping triggered")
            #     break
            
            # gc.collect()
        
        # Store fold results
        all_fold_results.append({
            'fold': fold_idx + 1,
            'best_f1': best_f1,
            'final_acc': fold_history['val_acc'][-1] if fold_history['val_acc'] else 0.0,
            'final_f1': fold_history['val_f1'][-1] if fold_history['val_f1'] else 0.0,
            'final_precision': fold_history['val_precision'][-1] if fold_history['val_precision'] else 0.0,
            'final_recall': fold_history['val_recall'][-1] if fold_history['val_recall'] else 0.0,
            'history': fold_history.copy()
        })
        
        print(f"  FOLD {fold_idx + 1} COMPLETE - Best F1: {best_f1:.4f}")
        
        # Cleanup   
        train_dataset.clear_cache()
        val_dataset.clear_cache()
        del model, optimizer, train_loader
        gc.collect()
            
    
        # Visualise results
        f1_scores = [r['best_f1'] for r in all_fold_results]
        acc_scores = [r['final_acc'] for r in all_fold_results]
        precision_scores = [r['final_precision'] for r in all_fold_results]
        recall_scores = [r['final_recall'] for r in all_fold_results]

        fig = plt.figure(figsize=(16, 20))
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fold_history['train_loss'], color='blue', label='Train Loss', markersize=4)
        plt.plot(fold_history['val_loss'], color='cyan', label='Val Loss', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('KAN Training & Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(fold_history['val_acc'], color='green', label='Val Accuracy', markersize=4)
        plt.plot(fold_history['val_f1'], color='magenta', label='Val F1', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('KAN Validation Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = project_root / 'deneme3_training_results2.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\nPlot saved to: {output_path}")
      