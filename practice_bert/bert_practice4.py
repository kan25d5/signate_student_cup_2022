""" 文書分類 （livedoorニュースコーパスによる実践文書分類） """
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

# from Utilities.const import JPMODEL_NAME
JPMODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"


# ------------------------------------------
# 定数
# ------------------------------------------
MAX_EPOCH_SIZE = 50
BATCH_SIZE = 100

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

CATEGORY_LIST = [
    "dokujo-tsushin",
    "it-life-hack",
    "kaden-channel",
    "livedoor-homme",
    "movie-enter",
    "peachy",
]

# ------------------------------------------
# データの前処理
# ------------------------------------------


def get_dataset_for_loader():
    import glob
    from tqdm import tqdm

    max_length = 128
    dataset_for_loader = []
    tokenizer = BertJapaneseTokenizer.from_pretrained(JPMODEL_NAME)

    # カテゴリごとにデータをロードして，dataset_for_loaderに追加する．
    for label, category in enumerate(tqdm(CATEGORY_LIST)):
        for file in glob.glob(f"/home/s2110184/dataset/text/{category}/*.txt"):
            # 行数ごとにリストにする
            lines = open(file).read().splitlines()
            # 4行目から抜き出す
            text = "\n".join(lines[:3])
            # 符号化
            encoding = tokenizer(
                text, max_length=max_length, padding="max_length", truncation=True
            )
            # ラベル情報を追加
            encoding["labels"] = label
            # GPUに飛ばす
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}
            # dataset_for_loaderに追加
            dataset_for_loader.append(encoding)

    # 確認
    print("execute get_dataset_for_loader()")
    print(dataset_for_loader[0])

    return dataset_for_loader


# ------------------------------------------
# データセットを分割する．
# ------------------------------------------


def get_dataloader(dataset_for_loader):
    import random
    from torch.utils.data import DataLoader

    random.shuffle(dataset_for_loader)

    train_size = int(TRAIN_RATIO * len(dataset_for_loader))
    val_size = int(VAL_RATIO * len(dataset_for_loader))

    dataset_train = dataset_for_loader[:train_size]
    dataset_val = dataset_for_loader[train_size : train_size + val_size]
    dataset_test = dataset_for_loader[train_size + val_size :]

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=256)
    dataloader_test = DataLoader(dataset_test, batch_size=256)

    print("execute get_dataloader()")
    for batch in dataloader_train:
        print("\t" + f"{batch}")
        break

    all_dataloader = [dataloader_train, dataloader_val, dataloader_test]
    return all_dataloader


# ------------------------------------------
# モデルの定義
# ------------------------------------------
import pytorch_lightning as pl


class BertForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, num_labels: int, lr=1e-5) -> None:
        super().__init__()
        self.lr = lr
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            JPMODEL_NAME, num_labels=num_labels
        )

    def training_step(self, batch, batch_idx):
        output = self.bert_sc.forward(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.bert_sc.forward(**batch)
        loss = output.loss
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        labels = batch.pop("labels")  # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct / labels.size(0)
        self.log("test_acc", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ------------------------------------------
# 学習コード
# ------------------------------------------


def training(all_dataloader, model):
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    dataloader_train = all_dataloader[0]
    dataloader_val = all_dataloader[1]
    dataloader_test = all_dataloader[2]

    # コールバックの定義
    check_point = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, save_weights_only=True, dirpath="pth/ldcc"
    )
    callbacks = [
        check_point,
        EarlyStopping(monitor="val_loss", patience=4),
    ]

    # Trainerの定義
    trainer = pl.Trainer(
        accelerator="gpu", devices=-1, max_epochs=MAX_EPOCH_SIZE, callbacks=callbacks
    )

    # ファインチューニングを行う
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    best_model_path = check_point.best_model_path
    best_model_score = check_point.best_model_score
    print("ベストモデルのファイル : {}".format(best_model_path))
    print("ベストモデルの損失スコア : {}".format(best_model_score))

    # テスト
    test = trainer.test(model, dataloader_test)
    print("Accuracy : {:.2f}".format(test[0]["accuracy"]))


# ------------------------------------------
# エントリーポイント
# ------------------------------------------


def main():
    dataset_for_loader = get_dataset_for_loader()
    all_dataloader = get_dataloader(dataset_for_loader)
    model = BertForSequenceClassification_pl(len(CATEGORY_LIST))

    training(all_dataloader, model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
