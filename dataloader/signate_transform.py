import re
import emoji
from nltk.corpus import stopwords


class SignateTransform(object):
    """
        SIGNATE STUDENT CUP 2022
        データセット用前処理クラス

        引数
        ------------
        - `is_remove_html` : bool = True \n
            HTMLタグを除去するか
        - `is_lower` : bool = True \n
            小文字化するか
        - `is_remove_stopwords` : bool = True \n 
            ストップワードを除去するか．
    """

    def __init__(self, is_remove_html=True, is_remove_stopwords=True, is_lower=True) -> None:
        self.is_remove_html = is_remove_html
        self.is_remove_stopwords = is_remove_stopwords
        self.is_lower = is_lower

        self.debris_chars = ["&amp;"]
        self.remove_patterns = [r"<\w+>", r"</\w+>"]
        self.replace_pattern_with_spaces = [r"(?s)<.*?>", r"&nbsp;"]
        self.replace_pair = {
            " </li><li> ": ". ",
            "</li><li>": ". ",
            "</li></ul>": ". ",
            "..": ". ",
            ". .": ". ",
        }
        self.debris_re = [re.compile(pattern) for pattern in self.remove_patterns]
        self.debris_sp = [re.compile(pattern) for pattern in self.replace_pattern_with_spaces]
        self.stop_words = frozenset(stopwords.words("english"))

    def _remove_html(self, text: str):
        for old, new in self.replace_pair.items():
            text = text.replace(old, new)
        for re_p in self.debris_re:
            text = re_p.sub("", text)
        for re_p in self.debris_sp:
            text = re_p.sub(" ", text)
        for debri in self.debris_chars:
            text = text.replace(debri, "")
        return text

    def _remove_stop_words(self, text):
        words = [word for word in text.split() if word not in self.stop_words]
        text = " ".join(words)
        return text

    def preprocess(self, text):
        """ 不要な文字列を除去する """

        # 文前後の空白を除去
        text = text.strip()
        # 顔文字を除去
        text = emoji.replace_emoji(text, replace="")

        if self.is_lower:
            text = text.lower()
        if self.is_remove_stopwords:
            text = self._remove_stop_words(text)
        if self.is_remove_html:
            text = self._remove_html(text)

        # 多重スペースを除去
        text = text.replace("  ", " ")
        text = text.replace("   ", " ")
        text = text.replace("    ", " ")

        return text

    def __call__(self, input_text):
        if type(input_text) == str:
            output = self.preprocess(input_text)
        elif type(input_text) == list:
            output = [self.preprocess(text) for text in input_text]
        else:
            raise TypeError("引数が許容するのはstr/listのみです．")
        return output


if __name__ == "__main__":
    import random
    import pandas as pd

    transform = SignateTransform()

    df = pd.read_csv("/home/s2110184/project/SIGNATE_Student_Cup_2022/assets/train.csv")

    for _ in range(5):
        index = random.randint(1, 1515)
        text = str(df.iloc[index, 1])

        print(f"index is {index}")
        print("before : ")
        print(text)
        print("after : ")
        print(transform(text))
        print("=" * 20)

