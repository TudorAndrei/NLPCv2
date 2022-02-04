import os

import numpy as np
import torch
from data_utils import TwitterDataset, Vocab
from model import Module, init_weights
from torch.nn.functional import one_hot
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 20
EMB_DIM = 64
HID_DIM = 128
NW = 8

# path_train = r"../twitter_dataset/train.csv"
path_train = r"../twitter_dataset/twitter_training.csv"
path_val = r"../twitter_dataset/twitter_validation.csv"
vocab = Vocab(path=path_train)
print("Created Vocab")

dataset_train = TwitterDataset(path=path_train, word2idx=vocab.vocab_dict)
dataset_val = TwitterDataset(path=path_val, word2idx=vocab.vocab_dict)
print("Created Dataset")

train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NW)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=NW)

num_embeddings = vocab.vocab_size + 1
model = Module(hidden_size=HID_DIM, num_embeddings=num_embeddings)

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())
model.cuda()

for epoch in range(EPOCHS):
    epoch_loss = 0
    acc = []
    model.train()
    for batch in train_loader:
        optim.zero_grad()
        tokens, label = batch["tokens"], batch["label"]
        tokens = tokens.permute(1, 0)
        label_OH = one_hot(label, num_classes=3).float()
        tokens = tokens.cuda()
        label_OH = label_OH.cuda()

        output = model(tokens)
        loss = criterion(output, label_OH)
        epoch_loss += loss.item()
        for i in range(output.shape[0]):
            acc.append(
                label[i].numpy() == np.argmax(output[i, :].cpu().detach().numpy())
            )
        loss.backward()
        optim.step()
    print(
        f"Epoch: {epoch+1}\nTrain: Loss: {epoch_loss/len(train_loader):.2f} Acc: {np.mean(np.array(acc)):.3f}"
    )
    model.eval()
    epoch_loss_val = 0
    acc = []
    for batch in val_loader:
        tokens, label = batch["tokens"], batch["label"]
        tokens = tokens.permute(1, 0)
        label_OH = one_hot(label, num_classes=3).float()
        tokens = tokens.cuda()
        label_OH = label_OH.cuda()

        output = model(tokens)
        loss_val = criterion(output, label_OH)
        epoch_loss_val += loss_val.item()
        for i in range(output.shape[0]):
            acc.append(
                label[i].numpy() == np.argmax(output[i, :].cpu().detach().numpy())
            )
    print(
        f"Val: Loss: {epoch_loss_val/len(train_loader):.2f} Acc: {np.mean(np.array(acc)):.3f}"
    )
os.system("notify-send 'Job Done ✔️'")
