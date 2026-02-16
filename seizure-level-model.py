from keras import layers, models, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import warnings

matplotlib.use('TkAgg')  # veya 'Agg'
warnings.filterwarnings("ignore")


def collect_npz_paths(root):
    return glob.glob(root + "/**/*.npz", recursive=True)


def load_npz_file(path):
    data = np.load(path)

    X = data['X']  # (epochs, channels, time)
    y = data['y']

    # CNN için transpose
    X = np.transpose(X, (0, 2, 1))  # -> (epochs, time, channels)

    return X, y



def robust_batch_generator(npz_paths, batch_size=32, buffer_size=2000):
    """
    Dosyaları bir havuzda (buffer) biriktirip karıştırarak batch üretir.
    Bu yöntem, Batch Normalization stabilitesi için KRİTİKTİR.

    Args:
        npz_paths: .npz dosya yolları listesi
        batch_size: Modelin bir seferde göreceği örnek sayısı
        buffer_size: Havuzda kaç örnek birikince shuffle yapılıp dağıtılacak
    """
    paths = npz_paths.copy()

    # Havuz (Buffer)
    buffer_X = []
    buffer_y = []

    while True:
        random.shuffle(paths)  # Dosya sırasını her epoch başında karıştır

        for path in paths:
            try:
                # Dosyayı yükle
                X_file, y_file = load_npz_file(path)

                # Eğer dosya boşsa veya hata varsa atla
                if len(X_file) == 0:
                    continue

                # Listeye ekle (Henüz numpy array yapmıyoruz, memory şişmesin)
                buffer_X.extend(X_file)
                buffer_y.extend(y_file)

                # Havuz doldu mu kontrol et
                while len(buffer_X) >= buffer_size:
                    # 1. Havuzu Numpy Array'e çevir
                    arr_X = np.array(buffer_X)
                    arr_y = np.array(buffer_y)

                    # 2. Havuzu Karıştır (Global Shuffle)
                    # Farklı dosyalardan gelen veriler birbirine girer -> Daha iyi eğitim
                    idx = np.arange(len(arr_X))
                    np.random.shuffle(idx)
                    arr_X = arr_X[idx]
                    arr_y = arr_y[idx]

                    # 3. Batch'leri Kes ve Yield Et
                    # Fazlalık kısmı (remainder) havuzda tutacağız
                    n_batches = len(arr_X) // batch_size

                    for i in range(n_batches):
                        start = i * batch_size
                        end = start + batch_size
                        yield arr_X[start:end], arr_y[start:end]

                    # 4. Kalanları (Remainder) Havuza Geri Koy
                    # Tam batch olmayan son parçayı atmıyoruz, sonraki dosyalardan gelenlerle birleşecek
                    remainder_start = n_batches * batch_size
                    buffer_X = list(arr_X[remainder_start:])
                    buffer_y = list(arr_y[remainder_start:])

                    # Bellek temizliği (önemli)
                    del arr_X, arr_y
                    # gc.collect() # Her döngüde çağırmak yavaşlatabilir, gerekirse açın

            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
                continue

def batch_generator(npz_paths, batch_size=32, shuffle_files=True):
    paths = npz_paths.copy()

    while True:

        if shuffle_files:
            random.shuffle(paths)

        for path in paths:

            X, y = load_npz_file(path)

            idx = np.arange(len(X))
            np.random.shuffle(idx)

            X = X[idx]
            y = y[idx]

            for i in range(0, len(X), batch_size):
                yield X[i:i + batch_size], y[i:i + batch_size]


def count_samples(npz_paths):
    total = 0

    for p in npz_paths:
        total += len(np.load(p)['y'])

    return total


def validation_generator(npz_paths):
    for path in npz_paths:
        X, y = load_npz_file(path)

        yield X, y


def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        focal = alpha * tf.pow(1 - p_t, gamma) * bce

        return tf.reduce_mean(focal)

    return loss


def f1_score(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return 2 * (precision * recall) / (precision + recall + 1e-7)


def build_eeg_cnn(input_shape):
    inp = layers.Input(shape=input_shape)

    # 1. Blok
    x = layers.Conv1D(64, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # 2. Blok
    x = layers.Conv1D(128, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # 3. Blok
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation='sigmoid')(x)  # Binary classification için doğru

    return models.Model(inp, out)


def plot_training_history(history):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    # ---- LOSS ----
    plt.figure()
    plt.plot(epochs, hist['loss'], label='Train Loss')
    plt.plot(epochs, hist['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # ---- ACCURACY ----
    plt.figure()
    plt.plot(epochs, hist['accuracy'], label='Train Acc')
    plt.plot(epochs, hist['val_accuracy'], label='Val Acc')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # ---- AUC ----
    plt.figure()
    plt.plot(epochs, hist['auc'], label='Train AUC')
    plt.plot(epochs, hist['val_auc'], label='Val AUC')
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()

    # ---- Precision ----
    plt.figure()
    plt.plot(epochs, hist['precision'], label='Train Precision')
    plt.plot(epochs, hist['val_precision'], label='Val Precision')
    plt.title("Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    # ---- Recall ----
    plt.figure()
    plt.plot(epochs, hist['recall'], label='Train Recall')
    plt.plot(epochs, hist['val_recall'], label='Val Recall')
    plt.title("Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.show()

    # ---- F1 ----
    plt.figure()
    plt.plot(epochs, hist['f1_score'], label='Train F1')
    plt.plot(epochs, hist['val_f1_score'], label='Val F1')
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.show()


def plot_all_metrics(history):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(12, 8))

    for key in hist.keys():
        if not key.startswith("val_"):
            plt.plot(epochs, hist[key], label=key)
            if "val_" + key in hist:
                plt.plot(epochs, hist["val_" + key], linestyle="--")

    plt.legend()
    plt.title("Training Metrics")
    plt.show()


#
# npz = np.load('dataset_final/train/chb01/chb01_03_train.npz')
# print(npz['X'].shape)
# print(np.count_nonzero(npz['y']))
#
# npz = np.load('dataset_final/val/chb01/chb01_03_val.npz')
# print(npz['X'].shape)
# print(np.count_nonzero(npz['y']))


if __name__ == '__main__':
    train_paths = collect_npz_paths("dataset-dummy/train")
    val_paths = collect_npz_paths("dataset-dummy/val")

    print("Train file count:", len(train_paths))
    print("Validation file count:", len(val_paths))

    BATCH_SIZE = 64
    BUFFER_SIZE = 4096

    train_gen = robust_batch_generator(train_paths, BATCH_SIZE, buffer_size=BUFFER_SIZE)
    val_gen = robust_batch_generator(val_paths, BATCH_SIZE, buffer_size=BUFFER_SIZE)

    total_train = count_samples(train_paths)
    total_val = count_samples(val_paths)

    steps_per_epoch = total_train // BATCH_SIZE
    val_steps = total_val // BATCH_SIZE

    # X, y = next(train_gen)
    # print(X.shape)
    # print(y.shape)
    # print("Train samples:", total_train)
    # print("Validation samples:", total_val)

    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=8,
        mode='max',
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        "best_eeg_model.keras",
        monitor='val_auc',
        mode='max',
        save_best_only=True
    )

    model = build_eeg_cnn((256, 18))

    model.compile(
        optimizer='adam',
        loss=focal_loss(),
        metrics=[
            'accuracy',
            metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            f1_score
        ]
    )

    model.summary()

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=50,
        callbacks=[
            early_stop,
            checkpoint,
            reduce_lr
        ]
    )

    print("*" * 60)
    plot_training_history(history)
    print("*" * 60)
    plot_all_metrics(history)