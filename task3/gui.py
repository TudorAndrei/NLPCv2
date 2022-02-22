import os
from tkinter import *

import torch
from all import (
    EncoderRNN,
    GreedySearchDecoder,
    LuongAttnDecoderRNN,
    evaluateInputChatbot,
    loadPrepareData,
)
from torch import nn

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell_movie"
corpus = os.path.join("..", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
model_name = "cb_model"
attn_model = "dot"

hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

checkpoint_iter = 4000
loadFilename = os.path.join(
    save_dir,
    model_name,
    corpus_name,
    "{}-{}_{}".format(encoder_n_layers, decoder_n_layers, hidden_size),
    "{}_checkpoint.tar".format(checkpoint_iter),
)


# If loading on same machine the model was trained on
checkpoint = torch.load(loadFilename)
# If loading a model trained on GPU to CPU
checkpoint = torch.load(loadFilename, map_location=torch.device("cpu"))
encoder_sd = checkpoint["en"]
decoder_sd = checkpoint["de"]
encoder_optimizer_sd = checkpoint["en_opt"]
decoder_optimizer_sd = checkpoint["de_opt"]
embedding_sd = checkpoint["embedding"]
voc.__dict__ = checkpoint["voc_dict"]


print("Building encoder and decoder ...")
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout
)
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print("Models built and ready to go!")
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()


# Example for validation

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)


def send():
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)

    if msg != "":

        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + "\n\n")
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = evaluateInputChatbot(msg, encoder, decoder, searcher, voc)
        ChatLog.insert(END, "Bot: " + res + "\n\n")

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(
    base,
    bd=0,
    bg="white",
    height=8,
    width=50,
    font="Arial",
)

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog["yscrollcommand"] = scrollbar.set

# Create Button to send message
SendButton = Button(
    base,
    font=("Verdana", 12, "bold"),
    text="Send",
    width="12",
    height=5,
    bd=0,
    bg="#32de97",
    activebackground="#3c9d9b",
    fg="#ffffff",
    command=send,
)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width=29, height=5, font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
