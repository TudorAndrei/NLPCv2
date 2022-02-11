import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import ChatbotDataset, Vocab
from model import Module
from torch.nn.functional import one_hot
from torch.utils.data.dataloader import DataLoader

BATCH_SIZE = 4
EPOCHS = 100
EMB_DIM = 32
HID_DIM = 32
NW = 8

path_train = r"../chatbot_intent/intent-corpus-basic.json"
# path_val = r"../chatbot_intent/intent-corpus-enrich-limit-20.json"
vocab = Vocab(path=path_train)
num_classes = len(vocab.label_encoder.keys())
print("Created Vocab")

dataset_train = ChatbotDataset(
    path=path_train,
    train=True,
    label_encoder=vocab.label_encoder,
    word2idx=vocab.vocab_dict,
    max_len=8,
)
dataset_val = ChatbotDataset(
    path=path_train,
    train=False,
    label_encoder=vocab.label_encoder,
    word2idx=vocab.vocab_dict,
    max_len=8,
)
print("Created Dataset")

train_loader = DataLoader(
    dataset_train, batch_size=BATCH_SIZE, num_workers=NW, drop_last=True
)
val_loader = DataLoader(
    dataset_val, batch_size=BATCH_SIZE, num_workers=NW, drop_last=True
)

num_embeddings = vocab.vocab_size + 1

# print(f"{num_classes=}")
# print(f"{num_embeddings=}")
# print(f"{num_classes=}")

model = Module(
    hidden_size=HID_DIM,
    num_embeddings=num_embeddings,
    num_classes=len(vocab.label_encoder.keys()),
    pad_idx=vocab.get_pad_idx(),
)

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())
# optim = torch.optim.SGD(model.parameters(), lr=0.01)
model.cuda()

train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_epoch = 1
best_loss = 999
for epoch in range(EPOCHS):
    epoch_loss = 0
    acc = []
    model.train()
    for batch in train_loader:
        optim.zero_grad()
        tokens, label = batch["tokens"], batch["label"]
        tokens = tokens.permute(1, 0)
        label_OH = one_hot(label, num_classes=num_classes).float()
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
    train_losses.append(epoch_loss / len(train_loader))
    train_accs.append(np.mean(np.array(acc)))
    # val
    model.eval()
    epoch_loss_val = 0
    acc_val = []
    for batch in val_loader:
        tokens, label = batch["tokens"], batch["label"]
        tokens = tokens.permute(1, 0)
        label_OH = one_hot(label, num_classes=num_classes).float()
        tokens = tokens.cuda()
        label_OH = label_OH.cuda()

        output = model(tokens)

        # print(output.shape)
        # print(label_OH.shape)
        loss_val = criterion(output, label_OH)
        epoch_loss_val += loss_val.item()
        for i in range(output.shape[0]):
            acc_val.append(
                label[i].numpy() == np.argmax(output[i, :].cpu().detach().numpy())
            )
    print(
        f"Val: Loss: {epoch_loss_val/len(val_loader):.2f} Acc: {np.mean(np.array(acc_val)):.3f}"
    )
    if (best_loss - (epoch_loss_val / len(val_loader))) > 0.1:
        best_epoch = epoch
        best_loss = epoch_loss_val / len(val_loader)
    val_losses.append(epoch_loss_val / len(val_loader))
    val_accs.append(np.mean(np.array(acc_val)))

os.system("notify-send 'Job Done ✔️'")


for layer in model.children():
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()

print(f"Best Epoch {best_epoch}")
for epoch in range(best_epoch):
    model.train()
    for batch in train_loader:
        optim.zero_grad()
        tokens, label = batch["tokens"], batch["label"]
        tokens = tokens.permute(1, 0)
        label_OH = one_hot(label, num_classes=num_classes).float()
        tokens = tokens.cuda()
        label_OH = label_OH.cuda()
        output = model(tokens)
        loss = criterion(output, label_OH)
        loss.backward()
        optim.step()

model.eval()
epoch_loss_val = 0
acc_val = []
for batch in val_loader:
    tokens, label = batch["tokens"], batch["label"]
    tokens = tokens.permute(1, 0)
    label_OH = one_hot(label, num_classes=num_classes).float()
    tokens = tokens.cuda()
    label_OH = label_OH.cuda()

    output = model(tokens)
    loss_val = criterion(output, label_OH)
    epoch_loss_val += loss_val
    for i in range(output.shape[0]):
        acc_val.append(
            label[i].numpy() == np.argmax(output[i, :].cpu().detach().numpy())
        )
print(
    f"Final Values:\nVal: Loss:\
    {epoch_loss_val/len(val_loader):.2f} Acc: {np.mean(np.array(acc_val)):.3f}"
)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
print(fig)
ax[0].plot(list(range(EPOCHS)), train_losses, label="Train")
ax[0].plot(list(range(EPOCHS)), val_losses, label="Val")
ax[0].axvline(x=best_epoch, color="red", linestyle="--")
ax[0].legend()
ax[0].title.set_text("Loss")

ax[1].plot(list(range(EPOCHS)), train_accs, label="Train")
ax[1].plot(list(range(EPOCHS)), val_accs, label="Val")
ax[1].axvline(x=best_epoch, color="red", linestyle="--")
ax[1].legend()
ax[1].title.set_text("Accuracy")
plt.show()
