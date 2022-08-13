import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import BertForSequenceClassification
from Utilities.const import ENMODEL_NAME


class SignateBert(pl.LightningModule):
    """
        SIGNATE STUDENT CUP 2022用
        BERTモデル
    """

    def __init__(self, output_dim: int, learning_ratio=1e-5):
        super().__init__()

        self.output_dim = output_dim
        self.lr = learning_ratio
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            ENMODEL_NAME, num_labels=self.output_dim
        )

    def forward(self, **kwargs):
        output = self.bert_sc(kwargs)
        return output

    def predict_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        return labels_predicted

    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        labels = batch.pop("labels")  # バッチからラベルを取得
        labels_predicted = self.predict_step(batch, batch_idx)
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct / labels.size(0)
        self.log("test_acc", accuracy)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
