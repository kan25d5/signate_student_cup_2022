""" 穴埋め問題を解く """


import torch
from transformers import BertForMaskedLM, BertJapaneseTokenizer
# from Utilities.const import JPMODEL_NAME
JPMODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"

tokenizer = BertJapaneseTokenizer.from_pretrained(JPMODEL_NAME)

# ------------------------------------------
# 準備
# ------------------------------------------

# 穴埋め問題を解くBERTモデルの定義
bert_mlm = BertForMaskedLM.from_pretrained(JPMODEL_NAME)
bert_mlm = bert_mlm.cuda()

# マスクの処理
text = "今日は[MASK]へ行く。"
tokens = tokenizer.tokenize(text)
print("# トークナイザ")
print("\t" + f"{tokens}")

# 符号化→GPUに載ってる
input_ids = tokenizer.encode(text, return_tensors="pt")
input_ids = input_ids.cuda()

# BERTに入力し，分類スコアを得る．
with torch.no_grad():
    output = bert_mlm(input_ids=input_ids)
    scores = output.logits

# ------------------------------------------
# 穴埋め問題を解く
# ------------------------------------------

# ID列で"[MASK]"（IDは4）の位置を調べる
mask_position = input_ids[0].tolist().index(4)

# スコアが最も良いトークンのIDを取り出し，トークンに変換する
id_best = scores[0, mask_position].argmax(-1).item()
token_best = tokenizer.convert_ids_to_tokens(id_best)
token_best = token_best.replace("##", "")

# "[MASK]"を上で求めたトークンに置換する
text = text.replace("[MASK]", token_best)

print(text)

# ------------------------------------------
# 上位10位の単語で置き換えた結果を出力する
# ------------------------------------------


def predict_mask_topk(text, tokeinzer, bert_mlm, num_topk):
    # 符号化
    input_ids = tokeinzer.encode(text, return_tensors="pt")
    input_ids = input_ids.cuda()

    # スコア算出
    with torch.no_grad():
        output = bert_mlm(input_ids=input_ids)
    scores = output.logits

    # スコアが上位のトークンとスコアを求める
    mask_position = input_ids[0].tolist().index(4)
    topk = scores[0, mask_position].topk(num_topk)
    ids_topk = topk.indices  # トークンのID
    tokens_topk = tokeinzer.convert_ids_to_tokens(ids_topk)
    scores_topk = topk.values.cpu().numpy()

    # 文章中の"[MASK]"を上で求めたトークンで置き換える
    text_topk = []
    for token in tokens_topk:
        token = token.replace("##", "")
        text_topk.append(text.replace("[MASK]", token, 1))

    return text_topk, scores_topk


text = "今日は[MASK]へ行く。"
text_topk, _ = predict_mask_topk(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep="\n")


# ------------------------------------------
# greedy_decodingによる方法
# ------------------------------------------
def greedy_prediction(text, tokeinzer, bert_mlm):
    for _ in range(text.count("[MASK]")):
        text = predict_mask_topk(text, tokeinzer, bert_mlm, 1)[0][0]
    return text


text = "今日は[MASK][MASK]へ行く。"
print("text : {}\npredict : {}".format(text, greedy_prediction(text, tokenizer, bert_mlm)))


# ------------------------------------------
# beam search による方法
# ------------------------------------------
