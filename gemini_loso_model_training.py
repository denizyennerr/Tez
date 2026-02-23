# %% Imports and Configuration
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

dataset_path = 'master_dataset_2s.npz'
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# %% Model Definition
def build_seizure_model(input_shape):
    model = models.Sequential([
        Input(shape=input_shape),
        # Block 1
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        # Block 2
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        # Block 3
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        # Classification Head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
    )
    return model


# %% Evaluation and Plotting Functions
def plot_fold_diagnostics(history, y_test, y_pred_prob, current_subject):
    """Generates a 2x2 dashboard of learning curves, CM, and ROC for DataSpell."""
    y_pred = (y_pred_prob > 0.5).astype(int)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Dashboard for Subject: {current_subject}', fontsize=16, fontweight='bold')

    # 1. Model Accuracy
    axs[0, 0].plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    axs[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    axs[0, 0].set_title('Model Accuracy')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Model Loss
    axs[0, 1].plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    axs[0, 1].plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    axs[0, 1].set_title('Model Loss (Binary Crossentropy)')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1, 0],
                xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'],
                annot_kws={"size": 14})
    axs[1, 0].set_title('Confusion Matrix')
    axs[1, 0].set_ylabel('True Label')
    axs[1, 0].set_xlabel('Predicted Label')

    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].set_title('Receiver Operating Characteristic (ROC)')
    axs[1, 1].legend(loc="lower right")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Ensure the suptitle doesn't overlap
    plt.show()


# %% Data Loading
data = np.load(dataset_path)
X, y, groups = data['X'], data['y'], data['s']

logo = LeaveOneGroupOut()
input_shape = (X.shape[1], X.shape[2])

print(f"Total Data Shape: {X.shape} | Subjects: {np.unique(groups)}")

# %% LOSO Training Loop
all_reports = []

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    current_test_subject = groups[test_idx][0]
    print(f"\n🚀 Training holding out Subject: {current_test_subject}")

    # Subject-level Validation Split
    unique_train_subjects = np.unique(groups_train)
    val_subject = unique_train_subjects[0]
    val_mask = (groups_train == val_subject)
    train_mask = ~val_mask

    X_train, X_val = X_train_full[train_mask], X_train_full[val_mask]
    y_train, y_val = y_train_full[train_mask], y_train_full[val_mask]

    model = build_seizure_model(input_shape)

    fold_log_dir = os.path.join(log_dir, f"subject_{current_test_subject}")

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        TensorBoard(log_dir=fold_log_dir, histogram_freq=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    # Rich DataFrame Reporting
    report_dict = classification_report(y_test, y_pred_class, target_names=['Normal', 'Seizure'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    print(f"\n📋 Classification Report for Subject {current_test_subject}:")
    display(report_df)

    all_reports.append({
        'subject': current_test_subject,
        'accuracy': report_dict['accuracy'],
        'seizure_f1': report_dict['Seizure']['f1-score'],
        'seizure_recall': report_dict['Seizure']['recall']
    })

    # Visualizations (Dashboard including Loss/Acc)
    plot_fold_diagnostics(history, y_test, y_pred_prob, current_test_subject)

    # break  # Uncomment to test just the first fold

# %% Aggregate Results Summary
summary_df = pd.DataFrame(all_reports)
print("\n🏆 Overall LOSO Summary:")
display(summary_df)
print(f"Mean Accuracy: {summary_df['accuracy'].mean():.4f}")
print(f"Mean Seizure Sensitivity (Recall): {summary_df['seizure_recall'].mean():.4f}")