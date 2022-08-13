""" BERTの順伝播 """


import torch
from transformers import BertJapaneseTokenizer, BertModel

# from Utilities.const import JPMODEL_NAME
JPMODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"


# --------------------------------------------------
# トークナイズ
# --------------------------------------------------
text = "明日は自然言語処理の勉強をしよう。"
tokenizer = BertJapaneseTokenizer.from_pretrained(JPMODEL_NAME)

# WordPieceによる分かち書き
print(tokenizer.tokenize("明日は自然言語処理の勉強しよう。"))
print(tokenizer.tokenize("明日はマシンラーニングの勉強しよう。"))

# 符号化
input_ids = tokenizer.encode(text)
print("input_ids : ")
print("\t" + f"{input_ids}")

# 復号化
convert_ids = tokenizer.convert_ids_to_tokens(input_ids)
print("convert_ids : ")
print("\t" + f"{convert_ids}")

# tokenizer
encoding = tokenizer(
    text, max_length=12, padding="max_length", truncation=True, return_tensors="pt"
)
print("# tokens : ")
print("\t" + f"{encoding}")
print("# tokens.keys() : ")
print("\t" + f"{[k for k in encoding.keys()]}")


# --------------------------------------------------
# BERTモデル
# --------------------------------------------------
text_list = ["明日は自然言語処理の勉強しよう。", "明日はマシンラーニングの勉強しよう。"]

# BERTモデルの定義
model = BertModel.from_pretrained(JPMODEL_NAME)
model = model.cuda()

print("# model.config")
print("\t" + f"{model.config}")

# トークナイザ
encoding = tokenizer(
    text_list, max_length=32, padding="max_length", truncation=True, return_tensors="pt"
)
# テンソルをGPUに載っける
encoding = {k: v.cuda() for k, v in encoding.items()}

# 順伝播
output = model(**encoding)
last_hidden_state = output.last_hidden_state
print("# last_hidden_state : ")
print("\t" + f"{last_hidden_state}")
print("\t" + f"{last_hidden_state.size()}")

# 順伝播（推論のみ）
with torch.no_grad():
    output = model(**encoding)
    last_hidden_state = output.last_hidden_state
