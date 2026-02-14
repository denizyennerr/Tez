import mne.io
from utility import model as yener
from utility import my_utils as deniz
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def run_loso_training(npz_dir, test_subject, use_zscore=True, save_best=True, plot_history=True):
    index = yener.get_npz_index(npz_dir)

    train_files, val_files = yener.loso_split(index, test_subject)

    print("Train files:", len(train_files))
    print("Val files:", len(val_files))

    if use_zscore:
        print("Computing Z-score stats...")
        mean, std = yener.compute_zscore_stats(train_files)

        train_gen = yener.normalized_batch_generator(train_files, mean, std)
        val_gen = yener.normalized_batch_generator(val_files, mean, std)

    else:
        train_gen = yener.batch_generator(train_files)
        val_gen = yener.batch_generator(val_files)

    model = yener.build_cnn_model()

    callbacks = []

    if save_best:
        best_model_path = f"best_model_test_{test_subject}.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"En iyi model {best_model_path} olarak kaydedilecek.")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=200,
        validation_steps=50,
        epochs=30,
        callbacks=callbacks if callbacks else None
    )

    if plot_history:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Eğitim Loss')
        plt.plot(history.history['val_loss'], label='Doğrulama Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
            plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
            plt.title('Accuracy')
        elif 'acc' in history.history:
            plt.plot(history.history['acc'], label='Eğitim Accuracy')
            plt.plot(history.history['val_acc'], label='Doğrulama Accuracy')
            plt.title('Accuracy')
        else:
            plt.text(0.5, 0.5, 'Accuracy metric yok', ha='center', va='center')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plot_path = f"training_history_test_{test_subject}.png"
        plt.savefig(plot_path)
        print(f"Eğitim grafikleri {plot_path} olarak kaydedildi.")
        plt.show()

    return model


if __name__ == "__main__":
    npz_dir = "dataset_v2"
    # Validation klasöründeki tüm denek isimlerini al
    test_subjects = deniz.get_folder_names('dataset_v2/val')
    print("Test edilecek denekler:", test_subjects)

    # Her bir denek için ayrı ayrı eğitim yap
    for subject in test_subjects:
        print(f"\n{'='*50}\nTest edilen denek: {subject}\n{'='*50}")
        model = run_loso_training(npz_dir=npz_dir, test_subject=subject, use_zscore=False)