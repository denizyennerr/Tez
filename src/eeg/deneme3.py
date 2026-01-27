"""
EEG Seizure Detection - CPU OPTIMIZED
Features:
1. Subject-Level Cross-Validation (GroupKFold Fixed)
2. RAM Efficiency: Contiguous Sampling (No cache thrashing)
3. Speed: Limits training to 16 batches per epoch as requested
"""
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- 1. Import Model ---
try:
    from eeg.models import KANSeizureDetector
except ImportError:
    current_file = Path(__file__).resolve()
    src_path = current_file.parent.parent.parent / 'src'
    sys.path.append(str(src_path))
    from eeg.models import KANSeizureDetector

# --- 2. Configuration ---
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
DATA_PATH = project_root / 'data' / 'preprocessed'

# -----------------------------------------------------------------------------
# 3. CRITICAL: IO-OPTIMIZED SAMPLER
# -----------------------------------------------------------------------------
class ContiguousBatchSampler(Sampler):
    """
    Yields batches from the same file consecutively.
    Essential for training on HDD/CPU to prevent cache thrashing.
    """
    def __init__(self, dataset, batch_size, max_batches=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_batches = max_batches
        
        # Group indices by file
        self.file_to_indices = defaultdict(list)
        for i, info in enumerate(dataset.index_map):
            self.file_to_indices[info['file_path']].append(i)
        self.file_paths = list(self.file_to_indices.keys())

    def __iter__(self):
        # Shuffle the order of files (e.g., File C -> File A -> File B)
        random.shuffle(self.file_paths)
        
        batch = []
        batches_yielded = 0
        
        for f_path in self.file_paths:
            indices = self.file_to_indices[f_path]
            # Shuffle indices WITHIN the file
            random.shuffle(indices)
            
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    batches_yielded += 1
                    
                    # STOP if we hit the debug limit
                    if self.max_batches and batches_yielded >= self.max_batches:
                        return
                        
        # Yield remaining if any (and limit allows)
        if batch and (not self.max_batches or batches_yielded < self.max_batches):
            yield batch

    def __len__(self):
        if self.max_batches:
            return self.max_batches
        return len(self.dataset) // self.batch_size

# -----------------------------------------------------------------------------
# 4. DATASET
# -----------------------------------------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, file_list, window_key='2s', max_cached_files=20):
        self.max_cached_files = max_cached_files
        self.index_map = []
        self._cache = OrderedDict()
        
        print(f"    Indexing {len(file_list)} files...")
        for f_path in file_list:
            try:
                # mmap_mode='r' lets us read headers without loading data to RAM
                with np.load(f_path, mmap_mode='r') as f: 
                    key = window_key if window_key in f else 'data'
                    if key not in f: continue
                    
                    labels = f['labels']
                    # Vectorized search for valid labels (0 or 1)
                    valid = np.where((labels == 0) | (labels == 1))[0]
                    
                    for i in valid:
                        self.index_map.append({
                            'file_path': str(f_path), 
                            'data_key': key, 
                            'index': int(i), 
                            'label': float(labels[i])
                        })
            except Exception:
                continue
            
    def __len__(self): return len(self.index_map)
    
    def _load_cache(self, path, key):
        if path in self._cache:
            self._cache.move_to_end(path)
            return
        
        # LRU Eviction
        while len(self._cache) >= self.max_cached_files:
            self._cache.popitem(last=False)
            
        with np.load(path, allow_pickle=True) as f:
            self._cache[path] = f[key][:].astype(np.float32)

    def __getitem__(self, idx):
        info = self.index_map[idx]
        self._load_cache(info['file_path'], info['data_key'])
        
        data = self._cache[info['file_path']][info['index']]
        if data.ndim > 1: data = data.flatten()
        
        # Norm
        m, s = data.mean(), data.std()
        if s > 1e-8: data = (data - m) / s
        
        return torch.from_numpy(data), torch.tensor([info['label']], dtype=torch.float32)
    
    def clear_cache(self):
        self._cache.clear()
        gc.collect()

    @staticmethod
    def get_patient_id(file_path: Path) -> str:
        """Extract patient ID from filename (e.g., chb01_03.npz -> chb01)."""
        return file_path.name.split('_')[0]

# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("KAN SEIZURE DETECTION - CPU OPTIMIZED CV")
    
    # --- CONFIG ---
    BATCH_SIZE = 16
    MAX_BATCHES_PER_EPOCH = 16  # Debug limit
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    PATIENCE = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Files
    all_files = list(DATA_PATH.glob("**/*.npz"))
    if not all_files:
        sys.exit("ERROR: No NPZ files found!")

    groups = [EEGDataset.get_patient_id(f) for f in all_files]
    unique_p = list(set(g for g in groups if g))
    print(f"Found {len(unique_p)} unique patients")
    
    # Cross Validation
    cv = GroupKFold(n_splits=min(5, len(unique_p)))
    all_fold_results = []

    # --- FIX: Iterate over split here to actually do Cross Validation ---
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(all_files, groups=groups)):
        print(f"\n{'='*40}\nFOLD {fold_idx + 1}/5\n{'='*40}")
        
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        
        # Verify Leakage
        tr_p = set(EEGDataset.get_patient_id(f) for f in train_files)
        va_p = set(EEGDataset.get_patient_id(f) for f in val_files)
        if tr_p & va_p: print(f"WARNING: Leakage detected {tr_p & va_p}")
        
        # Datasets
        train_ds = EEGDataset(train_files)
        val_ds = EEGDataset(val_files)
        
        # --- FIX: Use Contiguous Sampler for Training ---
        # Note: We pass MAX_BATCHES_PER_EPOCH to the sampler to enforce the limit efficiently
        train_sampler = ContiguousBatchSampler(train_ds, BATCH_SIZE, max_batches=MAX_BATCHES_PER_EPOCH)
        
        # shuffle=False because sampler handles it
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Calculate Class Weights (Approximate from indices)
        labels = [i['label'] for i in train_ds.index_map]
        neg, pos = labels.count(0.0), labels.count(1.0)
        pos_weight = neg / max(pos, 1) * 2.0
        print(f"  Class Balance: {neg} vs {pos} (Weight: {pos_weight:.2f})")

        # --- FIX: Init Model INSIDE loop ---
        x, _ = train_ds[0]
        model = KANSeizureDetector(
            input_dim=x.shape[0], 
            hidden_layers=[32, 16], 
            grid_size=3,
            dropout=0.3
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        
        # Training
        best_f1 = 0
        patience_counter = 0
        fold_history = {'train_loss': [], 'val_f1': [], 'val_acc': [], 'val_loss': []}

        for ep in range(NUM_EPOCHS):
            model.train()
            losses = []
            
            # Sampler handles the 'break' automatically
            pbar = tqdm(train_loader, desc=f"Ep {ep+1}", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            avg_trn_loss = np.mean(losses) if losses else 0
            
            # Validation
            model.eval()
            all_preds, all_targs = [], []
            val_batch_losses = []
            
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    if i >= MAX_BATCHES_PER_EPOCH:
                        break
                    
                    x = x.to(device)
                    y = y.to(device)  # Move labels to device for loss calculation
                    
                    out = model(x)
                    loss = criterion(out, y)
                    val_batch_losses.append(loss.item())

                    preds = (torch.sigmoid(out) > 0.5).float()
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_targs.extend(y.cpu().numpy().flatten())
            
            avg_val_loss = np.mean(val_batch_losses) if val_batch_losses else 0.0
            val_f1 = f1_score(all_targs, all_preds, zero_division=0)
            val_acc = accuracy_score(all_targs, all_preds)

            fold_history['train_loss'].append(avg_trn_loss)
            fold_history['val_loss'].append(avg_val_loss)
            fold_history['val_f1'].append(val_f1)
            fold_history['val_acc'].append(val_acc)

            # Early Stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), project_root / f'best_model_fold{fold_idx+1}.pth')
                msg = "Best F1"
            else:
                patience_counter += 1
                msg = "No Improvement"
                
            print(f"  Ep {ep+1}: Loss {avg_trn_loss:.4f} | F1 {val_f1:.4f} | Acc {val_acc:.4f} {msg}")
            
            if patience_counter >= PATIENCE:
                print("  Early Stopping")
                break
            
            gc.collect()

        # Store Results
        all_fold_results.append({
            'fold': fold_idx + 1,
            'best_f1': best_f1,
            'final_acc': fold_history['val_acc'][-1] if fold_history['val_acc'] else 0.0,
            'history': fold_history.copy()  # Store history for plotting
        })
        
        # Cleanup
        train_ds.clear_cache()
        val_ds.clear_cache()
        del model, optimizer, train_loader
        gc.collect()

    # Summary
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    f1s = [r['best_f1'] for r in all_fold_results]
    print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Visualise results (using last fold's history)
    if all_fold_results:
        last_fold = all_fold_results[-1]
        history = last_fold.get('history', {})
        
        plt.figure(figsize=(12, 5))

        # Plot Losses
        plt.subplot(1, 2, 1)
        if history.get('train_loss'):
            plt.plot(history['train_loss'], 'b-o', label='Train Loss', markersize=4)
        if history.get('val_loss'):
            plt.plot(history['val_loss'], 'r-o', label='Val Loss', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {last_fold["fold"]} - Training & Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot Accuracies and F1
        plt.subplot(1, 2, 2)
        if history.get('val_acc'):
            plt.plot(history['val_acc'], 'm-o', label='Val Acc', markersize=4)
        if history.get('val_f1'):
            plt.plot(history['val_f1'], 'orange', marker='o', linestyle='-', label='Val F1', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'Fold {last_fold["fold"]} - Validation Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = project_root / 'training_results3.png'
        plt.savefig(output_path)
        plt.close()
        print(f"\nPlot saved to: {output_path}")
    else:
        print("\nNo results to plot.")

