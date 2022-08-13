import torch
from transformers import BertTokenizer
from Utilities.const import ENMODEL_NAME, EN_LARGE_MODEL_NAME
from torch.utils.data import Dataset


class SignateDataset(Dataset):
    """
        SIGNATE STUDENT CUP 2022用
        Dataset クラス
    """

    def __init__(self, X, y=None, transform=None) -> None:
        self.X = X
        self.y = y
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained(ENMODEL_NAME)

        if self.y is not None:
            assert len(self.X) == len(self.y), "データとラベルのサイズが一致しない"

    def __getitem__(self, index: int):
        text = self.X[index]

        if self.transform is not None:
            text = self.transform(text)

        if self.y is not None:
            return {
                "text": text,
                "label": self.y[index],
            }
        else:
            return {
                "text": text,
                "label": [],
            }

    def __len__(self):
        return len(self.X)

    def collate_fn(self, batch):
        text_list = [item["text"] for item in batch]

        encoding = self.tokenizer(text_list, padding="longest", truncation=True)

        if self.y is not None:
            label_list = [item["label"] for item in batch]
            encoding["labels"] = label_list

        encoding = {k: torch.LongTensor(v) for k, v in encoding.items()}
        return encoding
