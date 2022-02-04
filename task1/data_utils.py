import pandas as pd
import numpy as np
import spacy
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer

label_encoder = {
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2,
    "Irrelevant": 2,
}

PAD_TOK = "PAD"
OOV_TOK = "OOV"


class Vocab:
    def __init__(self, path) -> None:
        self.vocab_dict = {PAD_TOK: 0, OOV_TOK: 1}
        self.generate_vocab(path)
        self.vocab_size = len(self.vocab_dict)

    def generate_vocab(self, path):
        data = pd.read_csv(path, header=None)
        nlp = spacy.load("en_core_web_sm")
        for line in range(len(data)):
            doc = nlp(str(data.iloc[line, 3]))
            for text in doc:
                if str(text) not in self.vocab_dict.keys() and text.is_alpha:
                    self.vocab_dict[str(text).lower()] = len(self.vocab_dict)


class TwitterDataset(Dataset):
    def __init__(self, path: str = None, word2idx={OOV_TOK: 1}, max_len=64) -> None:
        super().__init__()
        self.max_len = 512
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.data = pd.read_csv(path)
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad_tok = PAD_TOK

    def __getitem__(self, index):
        text = self.data.iloc[index, 3]
        tokens = []
        for t in self.tokenizer(str(text)):
            try:
                tokens.append(self.word2idx[t])
            except KeyError:
                tokens.append(self.word2idx[OOV_TOK])
        # pad tokens to max length
        if len(tokens) < self.max_len:
            for _ in range(self.max_len - len(tokens)):
                tokens.append(self.word2idx[self.pad_tok])
        else:
            tokens = tokens[: self.max_len]

        tokens = np.array(tokens)

        label = label_encoder[self.data.iloc[index, 2]]

        return {"tokens": tokens, "label": label}

    def __len__(self):
        return len(self.data)
