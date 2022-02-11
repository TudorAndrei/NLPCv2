import json

import numpy as np
import pandas as pd
import spacy
from pandas import DataFrame
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer

PAD_TOK = "PAD"
OOV_TOK = "OOV"


class Vocab:
    def __init__(self, path) -> None:
        self.path = path
        with open(self.path) as file:
            data = json.load(file)
        self.data = DataFrame.from_records(data["sentences"])
        self.vocab_dict = {PAD_TOK: 0, OOV_TOK: 1}
        self.label_encoder = {}
        self.generate_vocab()
        self.generate_label_ecoder()
        self.vocab_size = len(self.vocab_dict)

    def generate_vocab(self):
        data = self.data[self.data["training"] == True]
        nlp = spacy.load("en_core_web_sm")
        for line in range(len(data)):
            doc = nlp(str(data.iloc[line, 0]))
            for text in doc:
                if str(text) not in self.vocab_dict.keys() and text.is_alpha:
                    self.vocab_dict[str(text).lower()] = len(self.vocab_dict)

    def generate_label_ecoder(self):
        for i, element in enumerate(self.data["intent"].unique()):
            self.label_encoder[element] = i

    def get_pad_idx(self):
        return self.vocab_dict[PAD_TOK]


class ChatbotDataset(Dataset):
    def __init__(
        self, path, label_encoder, train, word2idx={OOV_TOK: 1}, max_len=8
    ) -> None:
        super().__init__()
        self.max_len = 512
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        with open(path) as file:
            data = json.load(file)
        data = DataFrame.from_records(data["sentences"])
        if train:
            self.data = data[data["training"] == True]
        else:
            self.data = data[data["training"] == False]
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad_tok = PAD_TOK
        self.label_encoder = label_encoder

    def __getitem__(self, index):
        text = self.data.iloc[index, 0]
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

        label = self.label_encoder[self.data.iloc[index, 1]]

        return {"tokens": tokens, "label": label}

    def __len__(self):
        return len(self.data)
