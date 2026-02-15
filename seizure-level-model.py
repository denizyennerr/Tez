import mne.io
from utility import model as yener
from utility import my_utils as deniz
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


# GEREKSİZ SATIRI KALDIRIN veya yorum satırı yapın:
# tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints', verbose=1, save_best_only=True)

def run_loso_training(npz_dir, test_subject, use_zscore=True, save_best=True, plot_history=True):
    index = yener.get_npz_index(npz_dir)

    train_files, val_files = yener.loso_split(index, test_subject)

    print("Train files:", len(train_files))
    print("Val files:", len(val_files))

    if use_zscore:
        print("Computing Z-score stats...")
        mean, std = yener.compute_zscore_stats(train_files)

        train_gen = yener.normalized_batch_generator_v2(train_files, mean, std)
        val_gen = yener.normalized_batch_generator_v2(val_files, mean, std)

    else:
        train_gen = yener.robust_batch_generator(train_files)
        val_gen = yener.robust_batch_generator(val_files)

    model = yener.build_cnn_model()

    callbacks = []

    if save_best:
        best_model_path = f"best_model_test_{test_subject}.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"En iyi model {best_model_path} olarak kaydedilecek.")

    # AFTER computing mean/std, ADD:
    total_train_segs = sum([np.load(f)['y'].shape[0] for f in train_files[:5]]) * max(1, len(train_files) // 5)
    steps_per_epoch = max(50, min(200, total_train_segs // 32))

    total_val_segs = sum([np.load(f)['y'].shape[0] for f in val_files[:2]]) * max(1,
                                                                                  len(val_files) // 2) if val_files else 10
    validation_steps = max(5, min(50, total_val_segs // 32))

    # THEN in model.fit():
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,  # ← DYNAMIC
        validation_steps=validation_steps,  # ← DYNAMIC
        epochs=10,
        callbacks=callbacks
    )

    if plot_history:
        plt.figure(figsize=(15, 4))

        # Loss subplot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], 'b-', label='Train loss')
        plt.plot(history.history['val_loss'], 'r-', label='Val loss')
        plt.title('Loss');
        plt.xlabel('Epoch');
        plt.legend();
        plt.grid(alpha=0.3)

        # AUC subplot (MOST IMPORTANT)
        plt.subplot(1, 3, 2)
        plt.plot(history.history['auc'], 'b-', label='Train Accuracy')
        plt.plot(history.history['val_auc'], 'r-', label='Validation Accuracy')
        plt.axhline(0.5, color='k', ls='--', alpha=0.3)
        plt.title('AUC');
        plt.xlabel('Epoch');
        plt.legend();
        plt.grid(alpha=0.3)

        # Precision/Recall
        plt.subplot(1, 3, 3)
        plt.plot(history.history['precision'], 'b-', label='Precision')
        plt.plot(history.history['recall'], 'r-', label='Recall')
        plt.title('Precision/Recall');
        plt.xlabel('Epoch');
        plt.legend();
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"training_history_{test_subject}.png")
        plt.show()

    return model


if __name__ == "__main__":
    npz_dir = "dataset_dummy"
    # Validation klasöründeki tüm denek isimlerini al
    test_subjects = deniz.get_folder_names('dataset_dummy/val')

    # Her bir denek için ayrı ayrı eğitim yap
    for subject in test_subjects:
        print(f"\n{'=' * 50}\nTest edilen denek: {subject}\n{'=' * 50}")
        model = run_loso_training(npz_dir=npz_dir, test_subject=subject, use_zscore=True)