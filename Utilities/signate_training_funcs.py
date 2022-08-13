def get_dataset(train_size=0.8, filepath="assets/train.csv", is_remove_stopwords=True):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from dataloader.signate_transform import SignateTransform
    from dataloader.signate_dataset import SignateDataset

    # データセットのロード
    df = pd.read_csv(filepath)

    # DataFrameをリスト化
    X = df["description"].to_list()
    y = df["jobflag"].to_list()

    # データの分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size)

    print(X_train[0])
    print(y_train[0])

    # DataSetを作成
    transform = SignateTransform(is_remove_stopwords=is_remove_stopwords)
    dataset_train = SignateDataset(X_train, y_train, transform)
    dataset_val = SignateDataset(X_val, y_val, transform)
    all_dataset = [dataset_train, dataset_val]

    return all_dataset


def get_dataloader(all_dataset, batch_size: int):
    from torch.utils.data import DataLoader

    dataloader_train = DataLoader(
        all_dataset[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=all_dataset[0].collate_fn,
    )
    dataloader_val = DataLoader(
        all_dataset[1],
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        collate_fn=all_dataset[1].collate_fn,
    )
    dataloader_test = DataLoader(
        all_dataset[1],
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        collate_fn=all_dataset[1].collate_fn,
    )

    all_dataloader = [dataloader_train, dataloader_val, dataloader_test]
    return all_dataloader
