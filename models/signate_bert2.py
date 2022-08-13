import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import F1Score
from transformers import BertModel
from Utilities.const import ENMODEL_HDIM, ENMODEL_NAME, EN_LARGE_MODEL_NAME, EN_LARGE_MODEL_HDIM


class SignateBert(pl.LightningModule):
    """
        SIGNATE STUDENT CUP 2022用
        BERTモデル
    """

    def __init__(self, output_dim: int, lr=1e-5, dropout1=0.2):
        # イニシャライザ
        super().__init__()

        # フィールド値
        self.output_dim = output_dim
        self.lr = lr
        self.bert_emb_dim = ENMODEL_HDIM

        # レイヤー定義
        self.bert = BertModel.from_pretrained(ENMODEL_NAME)
        self.dropout1 = nn.Dropout(dropout1)
        self.out = nn.Linear(self.bert_emb_dim, output_dim)

        # 損失関数，評価関数の定義
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.f1 = F1Score(output_dim, average="macro")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """ 順伝播 """
        bert_out = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        x = bert_out["last_hidden_state"][:, 0, :]
        x = self.dropout1(x)
        x = self.out(x)
        return x

    def compute_loss(self, preds, labels):
        """ 損失関数 """
        preds = preds.reshape(-1, preds.shape[-1])
        labels = labels.reshape(-1)
        loss = self.criterion(preds, labels)
        return loss

    def predict_step(self, batch, batch_idx):
        """ 推論 """
        self.eval()
        output = self(**batch)
        labels_predicted = output.argmax(-1)
        return labels_predicted

    def training_step(self, batch, batch_idx):
        """ バッチ学習 """
        self.train()
        labels = batch.pop("labels")
        preds = self(**batch)
        loss = self.compute_loss(preds, labels)
        self.log("train_loss", value=loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """ バッチ検証 """
        labels = batch.pop("labels")
        preds = self(**batch)
        loss = self.compute_loss(preds, labels)
        labels_predicted = preds.argmax(-1)
        self.log("val_loss", value=loss)
        result = {"labels": labels, "labels_predicted": labels_predicted}
        return result

    def validation_epoch_end(self, validation_step_outputs):
        """ エポック毎のF1スコア算出 """
        labels = []
        labels_predicted = []

        for out in validation_step_outputs:
            labels.extend(out["labels"])
            labels_predicted.extend(out["labels_predicted"])

        labels = torch.tensor(labels, device=self.device)
        labels_predicted = torch.tensor(labels_predicted, device=self.device)

        self.val_f1 = self.f1(labels, labels_predicted)
        self.log("val_f1", value=self.val_f1)

    def test_step(self, batch, batch_idx):
        """ バッチテスト """
        self.eval()
        labels = batch.pop("labels")
        labels_predicted = self.predict_step(batch, batch_idx)
        result = {
            "labels": labels,
            "labels_predicted": labels_predicted,
        }
        return result

    def test_epoch_end(self, test_step_outputs):
        """ テストデータのF1スコア算出 """
        labels = []
        labels_predicted = []

        for out in test_step_outputs:
            labels.extend(out["labels"])
            labels_predicted.extend(out["labels_predicted"])

        labels = torch.tensor(labels, device=self.device)
        labels_predicted = torch.tensor(labels_predicted, device=self.device)

        test_f1 = self.f1(labels, labels_predicted)
        self.log("test_f1", value=test_f1)

    def configure_optimizers(self):
        """ 最適化関数 """
        return optim.Adam(self.parameters(), lr=self.lr)
