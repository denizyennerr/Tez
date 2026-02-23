import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib
# matplotlib.use('TkAgg')

dataset_path= 'master_dataset_2s.npz'

def build_seizure_model(input_shape):
    model = models.Sequential([
        Input(shape=input_shape),
        # 1. Katman
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        # 2. Katman
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        # 3. Katman
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        # Sınıflandırma
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

# --- 2. GÖRSELLEŞTİRME FONKSİYONLARI ---
def plot_results(history, y_test, y_pred_prob, current_subject):
    y_pred = (y_pred_prob > 0.5).astype(int)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Test Edilen Hasta: {current_subject}', fontsize=16)

    # 1. Accuracy
    axs[0, 0].plot(history.history['accuracy'], label='Train')
    axs[0, 0].plot(history.history['val_accuracy'], label='Val')
    axs[0, 0].set_title('Model Accuracy')
    axs[0, 0].legend()

    # 2. Loss
    axs[0, 1].plot(history.history['loss'], label='Train')
    axs[0, 1].plot(history.history['val_loss'], label='Val')
    axs[0, 1].set_title('Model Loss')
    axs[0, 1].legend()

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1, 0],
                xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'])
    axs[1, 0].set_title('Confusion Matrix')

    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axs[1, 1].plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
    axs[1, 1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axs[1, 1].set_title('ROC Curve')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 3. VERİ YÜKLEME VE LOSO DÖNGÜSÜ ---

data = np.load(dataset_path)
X, y, groups = data['X'], data['y'], data['s']

logo = LeaveOneGroupOut()
input_shape = (X.shape[1], X.shape[2])

print(f"Toplam Veri: {X.shape} | Gruplar (Hastalar): {np.unique(groups)}")

for train_idx, test_idx in logo.split(X, y, groups=groups):
    # Test setini ayır
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    current_test_subject = groups[test_idx][0]
    print(f"\n🚀 Şu an Test Edilen Hasta: {current_test_subject}")

    # --- Subject-level Validation Split ---
    # Eğitim setindeki hastaların birini validation için ayırıyoruz
    unique_train_subjects = np.unique(groups_train)
    val_subject = unique_train_subjects[0]  # İlk hastayı val yap

    val_mask = (groups_train == val_subject)
    train_mask = ~val_mask

    X_train, X_val = X_train_full[train_mask], X_train_full[val_mask]
    y_train, y_val = y_train_full[train_mask], y_train_full[val_mask]
    # İlk hastadan sonra durmak istersen (test için):
    # break
    # --- Modeli Eğit ---
    model = build_seizure_model(input_shape)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # --- Tahmin ve Değerlendirme ---
    y_pred_prob = model.predict(X_test)

    # Rapor ve Görselleştirme
    print(f"\n📋 Classification Report for Subject {current_test_subject}:")
    print(classification_report(y_test, (y_pred_prob > 0.5).astype(int), target_names=['Normal', 'Seizure']))

    plot_results(history, y_test, y_pred_prob, current_test_subject)

