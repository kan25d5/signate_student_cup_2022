import random
import warnings

import numpy as np
import torch
import optuna

# ワーニング
warnings.simplefilter("ignore")

# 定数
SEED = 0

MAX_EPOCH_SIZE = 100
BATCH_SIZE = 40

# トレーニングチューニング
N_TRIALS = 200
PATIENCE = 2

CATEGORY_SIZE = 5
ACCELERATOR = "gpu"
DEVICES = 2


# 乱数シードの設定
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# グローバル変数
pre_best_score = 0.0


def training(lr, batch_size, train_size, dropout1, trial=None):
    """トレーニング関数"""

    global pre_best_score

    # ----------------------------------------
    # DataLoder作成
    # ----------------------------------------
    from Utilities.signate_training_funcs import get_dataset, get_dataloader

    all_dataset = get_dataset(train_size, is_remove_stopwords=False)
    all_dataloader = get_dataloader(all_dataset, batch_size)

    # ----------------------------------------
    # モデルの定義
    # ----------------------------------------
    from models.signate_bert2 import SignateBert

    model = SignateBert(CATEGORY_SIZE, lr=lr, dropout1=dropout1)

    # ----------------------------------------
    # Trainerの定義
    # ----------------------------------------
    import os
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
    from pytorch_lightning.loggers import TensorBoardLogger

    if trial is not None:
        trainer = pl.Trainer(
            accelerator=ACCELERATOR,
            devices=DEVICES,
            max_epochs=MAX_EPOCH_SIZE,
            callbacks=[
                EarlyStopping(monitor="val_f1", verbose=True, patience=PATIENCE, mode="max"),
                PyTorchLightningPruningCallback(trial, monitor="val_f1"),
            ],
            logger=TensorBoardLogger(os.getcwd(), version=f"trial:{trial.number}"),
        )
    else:
        trainer = pl.Trainer(
            accelerator=ACCELERATOR,
            devices=DEVICES,
            max_epochs=MAX_EPOCH_SIZE,
            callbacks=[
                EarlyStopping(monitor="val_f1", verbose=True, patience=PATIENCE, mode="max"),
            ],
        )

    # ----------------------------------------
    # 訓練
    # ----------------------------------------
    trainer.fit(model, train_dataloaders=all_dataloader[0], val_dataloaders=all_dataloader[1])
    trainer.test(model, all_dataloader[2])

    # ----------------------------------------
    # test_dataloaderのF1スコアを返す
    # ----------------------------------------
    test_score = trainer.callback_metrics["test_f1"].item()

    if test_score > pre_best_score:
        pre_best_score = test_score
        predict(model)
        torch.save(model.state_dict(), "pth/signate/best_model.pth")

    return test_score


def predict(model):
    import pandas as pd
    from dataloader.signate_dataset import SignateDataset
    from dataloader.signate_transform import SignateTransform
    from torch.utils.data import DataLoader

    # ----------------------------------------
    # submit用（test.csv）のデータローダーを作成
    # ----------------------------------------
    df = pd.read_csv("assets/test.csv")
    X_test = df["description"].to_list()
    submit_dataset = SignateDataset(X_test, transform=SignateTransform())
    dataloader_submit = DataLoader(
        submit_dataset, batch_size=1, shuffle=False, collate_fn=submit_dataset.collate_fn
    )

    # ----------------------------------------
    # 推論
    # ----------------------------------------
    output_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader_submit):
            labels_predicted = model.predict_step(batch, batch_idx)
            output_list.append(labels_predicted.item())

    start_idx = 1516
    with open("assets/submit.csv", "w") as f:
        for output in output_list:
            f.write(f"{start_idx},{output}\n")
            start_idx += 1


def objective(trial: optuna.Trial):
    """目的関数"""

    lr = trial.suggest_float("lr", 1e-8, 1e-4)
    dropout1 = trial.suggest_float("dropout1", 0.3, 0.7)

    train_size = 0.8
    batch_size = 28

    print("ハイパラ情報 : ")
    print("\tlr : " + f"{lr}")
    print("\tdropout1 : " + f"{dropout1}")

    return training(lr, batch_size, train_size, dropout1, trial=trial)


def main():
    import optuna

    pruner = optuna.pruners.MedianPruner(n_startup_trials=int(N_TRIALS * 0.3), n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    training(lr=4.44e-05, batch_size=27, train_size=0.811, dropout1=0.4528)
