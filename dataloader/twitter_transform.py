import re
import emoji
import neologdn
import unicodedata
from fugashi import Tagger

RE_PATTERNS = [r"#[^#\s]*", r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", r"@[a-zA-Z\d_]+"]
REMOVE_CHARS = ["�"]


class TwitterTransform(object):
    def __init__(self) -> None:
        self.tagger = Tagger("-Owakati")
        self.re_removes = [re.compile(pattern) for pattern in RE_PATTERNS]

    def __call__(self, text: str):
        for rc in REMOVE_CHARS:
            text = text.replace(rc, "")
        for re_remove in self.re_removes:
            text = re_remove.sub("", text)

        text = text.strip()
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        text = neologdn.normalize(text)
        text = emoji.replace_emoji(text, replace="")

        wakati = self.tagger.parse(text)
        return wakati
