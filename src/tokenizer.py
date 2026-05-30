import re
import os
import unicodedata
from collections import Counter
from itertools import chain
import joblib

class Tokenizer:
    SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|sayi|>", "<|sayi_araligi|>", "<|tarih|>"]

    def __init__(self, min_count=5) -> None:
        self.min_count = min_count
        self.counter: Counter
        self.word2idx = {}
        self.idx2word = {}

    def clean_text(self, text):
        text = unicodedata.normalize("NFC", text)
        text = text.lower().replace("i\u0307", "i")
        aylar = "ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık"

        # "18 ağustos 1227" gibi tarihleri tek tokena çevirir
        text = re.sub(rf"\b\d{{1,2}}\s+({aylar})\s+\d{{3,4}}\b", " <|tarih|> ", text)

        # "1206-1227", "1914 – 1918" gibi sayı aralıklarını tek tokena çevirir
        text = re.sub(r"\b\d+\s*[-–—]\s*\d+\b", " <|sayi_araligi|> ", text)

        # "2023te", "000inden", "5inci" gibi sayı+ek yapılarını sayı tokenı ve ek olarak ayırır
        text = re.sub(r"\b(\d+)([a-zçğıöşü]+)\b", r" <|sayi|> \2 ", text)

        # kalan tekil sayıları sayı tokenına çevirir
        text = re.sub(r"\b\d+\b", " <|sayi|> ", text)

        # apostrof ve tırnak işaretlerini siler, kelimeleri bölmez
        text = re.sub(r"[\u0027\u0022\u0060\u2019\u2018\u201C\u201D]", "", text)

        # kelime arasındaki tire dahil tüm tireleri boşluğa çevirir
        text = re.sub(r"[-–—]", " ", text)

        # harf, sayı, alt çizgi, boşluk ve özel token karakterleri dışındaki noktalama işaretlerini temizler
        text = re.sub(r"[^\w\s<>|çğıöşü]", " ", text)

        # fazla boşlukları teke indirir
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def tokenize(self, text):
        return self.clean_text(text).split()

    def _transform(self, corpus):

        def process_batch(batch):
            cleaned = [self.clean_text(text) for text in batch["text"]]
            tokens = [text.split() for text in cleaned]
            return {"clean_text": cleaned, "tokens": tokens}

        return corpus.map(process_batch, batched=True, batch_size=1000, num_proc=os.cpu_count())

    def fit(self, corpus):
        tokenized_ds = self._transform(corpus)

        counter = Counter(chain.from_iterable(tokenized_ds["tokens"]))

        vocab_words = [w for w, count in counter.items() if count >= self.min_count and w not in self.SPECIAL_TOKENS]
        vocab = self.SPECIAL_TOKENS + vocab_words

        self.idx2word = vocab
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

        return tokenized_ds["tokens"], counter

    def index(self, word):
        return self.word2idx.get(word, self.word2idx["<|unk|>"])

    def word(self, idx):
        return self.idx2word[idx]

    def encode(self, tokens):

        if isinstance(tokens[0], str):
            return [self.index(word) for word in tokens]
    
        return [
            [self.index(word) for word in sentence]
            for sentence in tokens
        ]
    
    def encode_text(self, text):
        return self.encode(self.tokenize(text))

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return None

        return joblib.load(path)

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        joblib.dump(self, path)