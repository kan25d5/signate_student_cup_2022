""" 文書分類（例題） """


import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
# from Utilities.const import JPMODEL_NAME
JPMODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"

# ------------------------------------------
# 準備
# ------------------------------------------

# トークナイザ
tokenizer = BertJapaneseTokenizer.from_pretrained(JPMODEL_NAME)
# 分類BERTモデル
bert_sc = BertForSequenceClassification.from_pretrained(JPMODEL_NAME, num_labels=2)
# GPUに飛ばす
bert_sc = bert_sc.cuda()

# ------------------------------------------
# 例題
# ------------------------------------------

# データ
text_list = [
    "この映画は面白かった。",
    "この映画の最後にはがっかりさせられた。",
    "この映画を見て幸せな気持ちになった。",
]
label_list = [1, 0, 1]

# データの符号化
encoding = tokenizer(text_list, padding="longest", return_tensors="pt")
encoding = {k: v.cuda() for k, v in encoding.items()}
labels = torch.tensor(label_list).cuda()

# 推論
with torch.no_grad():
    output = bert_sc.forward(**encoding)
scores = output.logits
labels_predicted = scores.argmax(-1)
num_correct = (labels_predicted == labels).sum().item()
accuracy = num_correct / labels.size(0)

print("# scoresのサイズ：")
print("\t" + f"{scores.size()}")
print("# predicted labels：")
print("\t" + f"{labels_predicted}")
print("# target labels：")
print("\t" + f"{label_list}")
print("# accuracy : ")
print("\t" + f"{accuracy}")

