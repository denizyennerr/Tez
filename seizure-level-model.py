from utility import model as yener
from utility import my_utils as deniz

npz_dir = "dataset_dummy"

# yener.report_folder_sizes('dataset/train')
index = yener.get_npz_index("dataset_dummy")

test_subject = deniz.get_folder_names('dataset_dummy/val')

train_files, val_files = yener.loso_split(index, test_subject)
print(train_files)
print(val_files)


def run_loso_training(npz_dir, test_subject, use_zscore=True):
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

    model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=200,
        validation_steps=50,
        epochs=30
    )

    return model


if __name__ == "__main__":
    npz_dir = "dataset_dummy"
    # yener.report_folder_sizes('dataset/train')
    index = yener.get_npz_index("dataset_dummy")

    test_subject = deniz.get_folder_names('dataset_dummy/val')

    run_loso_training(npz_dir=npz_dir, test_subject=test_subject, use_zscore=False)
    model=run_loso_training(npz_dir=npz_dir, test_subject=test_subject, use_zscore=False)


