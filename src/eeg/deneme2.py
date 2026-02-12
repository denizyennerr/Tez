"""
EEG Seizure Detection Training Script - CROSS-VALIDATION CORRECTED
Implements proper K-fold cross-validation with per-fold training
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    """Memory-efficient EEG dataset with file caching."""
    
    def __init__(self, file_list: list, window_size_key: str = '2s', max_cached_files: int = 5):
        self.file_list = file_list
        self.window_size_key = window_size_key
        self.max_cached_files = max_cached_files
        self.index_map = []
        
        self._cache = {}
        self._cache_order = []
        
        print(f"    Building index map from {len(file_list)} files...")
        for f_path in file_list:
            try:
                with np.load(f_path, allow_pickle=True) as f:
                    key = self.window_size_key if self.window_size_key in f else 'data'
                    if key not in f or 'labels' not in f:
                        continue
                    
                    n_samples = f[key].shape[0]
                    labels = f['labels']
                    
                    for i in range(n_samples):
                        if labels[i] in (0, 1):
                            self.index_map.append({
                                'file_path': str(f_path),
                                'data_key': key,
                                'index': i,
                                'label': float(labels[i])
                            })
            except Exception as e:
                print(f"    Warning: Could not load {f_path.name}: {e}")
                continue
        
        print(f"    Index map: {len(self.index_map)} samples")

    def __len__(self):
        return len(self.index_map)
    
    def _load_file_to_cache(self, file_path: str, data_key: str):
        if file_path in self._cache:
            if file_path in self._cache_order:
                self._cache_order.remove(file_path)
            self._cache_order.append(file_path)
            return
        
        while len(self._cache) >= self.max_cached_files and self._cache_order:
            oldest = self._cache_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
            # Note: gc.collect() removed - called once per epoch instead
        
        with np.load(file_path, allow_pickle=True) as f:
            self._cache[file_path] = {
                'data': f[data_key][:].astype(np.float32),
                'labels': f['labels'][:].astype(np.float32)
            }
        self._cache_order.append(file_path)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range")
        
        info = self.index_map[idx]
        file_path = info['file_path']
        data_key = info['data_key']
        sample_idx = info['index']
        
        self._load_file_to_cache(file_path, data_key)
        
        data = self._cache[file_path]['data'][sample_idx]
        label = info['label']
        
        if data.ndim > 1:
            data = data.flatten()
        
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-8:
            data = (data - mean) / std
        
        return (
            torch.from_numpy(data.copy()),
            torch.tensor([label], dtype=torch.float32)
        )
    
    def clear_cache(self):
        self._cache.clear()
        self._cache_order.clear()
        gc.collect()
    
    @staticmethod
    def get_patient_id(file_path: Path) -> str:
        name = file_path.name
        if 'chb' in name.lower():
            parts = name.split('_')
            return parts[0] if parts else name
        return name.split('_')[0] if '_' in name else name


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("KAN SEIZURE DETECTION - K-Fold Cross-Validation Training")
    print("=" * 70)
    
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
    
    groups = [EEGDataset.get_patient_id(f) for f in all_files]
    unique_patients = list(set(g for g in groups if g is not None))
    print(f"Found {len(unique_patients)} unique patients")
    
    
    # -------------------------------------------------------------------------
    # CROSS-VALIDATION SETUP
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
    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False, drop_last=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    print("\nInitializing model...")    
    sample_x, sample_y = train_dataset[0]
    input_dim = sample_x.shape[0]
    print(f"\nInput dimension: {input_dim}")

    model = KANSeizureDetector(
        input_dim=input_dim,
        hidden_layers=[32, 16],
        grid_size=3,
        dropout=0.3 
    )
    model.to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # -------------------------------------------------------------------------
    # LOSS AND OPTIMIZER
    # -------------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training: {NUM_EPOCHS} epochs, {MAX_BATCHES_PER_EPOCH} batches/epoch")
    print(f"{'='*60}")
    
    all_fold_results = []
    
    for fold_idx in range(5):  # 5 folds
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{5}")
        print(f"{'='*60}")  
        
        # Training loop for this fold
        print(f"\n  {'='*68}")
        print(f"  TRAINING FOLD {fold_idx + 1}")
        print(f"  {'='*68}\n")
        
        best_val_f1 = 0.0
        patience_counter = 0

        batch_losses = []
        fold_history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }

        for epoch in range(NUM_EPOCHS):
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
                del X_batch, y_batch, y_pred, loss
            
            avg_train_loss = np.mean(batch_losses)
            fold_history['train_loss'].append(avg_train_loss.item())
            
            # Memory cleanup
            del batch_losses
            gc.collect()
            
            # Validation phase
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
            
            avg_val_loss = np.mean(val_batch_losses)
            val_acc = accuracy_score(all_targets, all_preds)
            val_f1 = f1_score(all_targets, all_preds, zero_division=0)
            val_precision = precision_score(all_targets, all_preds, zero_division=0)
            val_recall = recall_score(all_targets, all_preds, zero_division=0)
            
            fold_history['val_loss'].append(avg_val_loss.item())
            fold_history['val_acc'].append(val_acc)
            fold_history['val_f1'].append(val_f1.item())
            fold_history['val_precision'].append(val_precision.item())
            fold_history['val_recall'].append(val_recall.item())
            
            scheduler.step()
            

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(
                    model.state_dict(), 
                    project_root / f'best_model_fold{fold_idx+1}.pth'
                )
                improvement_flag = "✓ NEW BEST"
            else:
                patience_counter += 1
                improvement_flag = f"(no improvement: {patience_counter}/{NUM_EPOCHS})"
            
            print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"F1: {val_f1:.4f} | {improvement_flag}")
            
            if patience_counter >= NUM_EPOCHS:
                print(f"  Early stopping at epoch {epoch+1}")
                break
            
            gc.collect()
        
        # Store fold results
        fold_results = {
            'fold': fold_idx + 1,
            'best_f1': best_val_f1,
            'final_acc': fold_history['val_acc'][-1],
            'final_f1': fold_history['val_f1'][-1],
            'final_precision': fold_history['val_precision'][-1],
            'final_recall': fold_history['val_recall'][-1],
            'history': fold_history,
            'train_patients': sorted(train_patients),
            'val_patients': sorted(val_patients)
        }
        all_fold_results.append(fold_results)
        
        print(f"\n  {'='*68}")
        print(f"  FOLD {fold_idx + 1} COMPLETE - Best F1: {best_val_f1:.4f}")
        print(f"  {'='*68}\n")
        
        # Cleanup
        train_dataset.clear_cache()
        val_dataset.clear_cache()
        del model, optimizer, scheduler, criterion
        del train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
    
    # -------------------------------------------------------------------------
    # CROSS-VALIDATION SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    
    f1_scores = [r['best_f1'] for r in all_fold_results]
    acc_scores = [r['final_acc'] for r in all_fold_results]
    precision_scores = [r['final_precision'] for r in all_fold_results]
    recall_scores = [r['final_recall'] for r in all_fold_results]
    
    print(f"\nPer-Fold Results:")
    for r in all_fold_results:
        print(f"  Fold {r['fold']}: F1={r['best_f1']:.4f}, Acc={r['final_acc']:.4f}, "
              f"Precision={r['final_precision']:.4f}, Recall={r['final_recall']:.4f}")
    
    print(f"\nAveraged Metrics (Mean ± Std):")
    print(f"  F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"  Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"  Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"  Recall:    {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    
    # -------------------------------------------------------------------------
    # SAVE COMPREHENSIVE PLOTS
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: F1 scores per fold (box plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.boxplot([f1_scores], labels=['F1 Score'])
    ax1.scatter([1]*len(f1_scores), f1_scores, alpha=0.5, c='red')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score Distribution Across Folds')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All metrics per fold (bar plot)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(all_fold_results))
    width = 0.2
    ax2.bar(x - 1.5*width, f1_scores, width, label='F1', alpha=0.8)
    ax2.bar(x - 0.5*width, acc_scores, width, label='Accuracy', alpha=0.8)
    ax2.bar(x + 0.5*width, precision_scores, width, label='Precision', alpha=0.8)
    ax2.bar(x + 1.5*width, recall_scores, width, label='Recall', alpha=0.8)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Score')
    ax2.set_title('Metrics Per Fold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"F{i+1}" for i in range(len(all_fold_results))])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3-6: Training curves for each fold
    for i, result in enumerate(all_fold_results[:4]):  # Max 4 folds
        ax = fig.add_subplot(gs[1 + i//2, i%2])
        history = result['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f"Fold {result['fold']} Training Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'K-Fold Cross-Validation Results (Mean F1: {np.mean(f1_scores):.4f})', 
                 fontsize=14, fontweight='bold')
    
    output_path = project_root / 'cv_results_complete.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION COMPLETE!")
    print(f"{'='*70}")
