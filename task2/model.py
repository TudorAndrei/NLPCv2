import torch


class Module(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_embeddings,
        num_classes=3,
        embedding_dim=32,
        num_layers=1,
        bidirectional=False,
        dropout_p=0.2,
        pad_idx=None,
    ):
        super().__init__()

        self.emb = torch.nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            padding_idx=pad_idx,
        )

        self.rnn = torch.nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        rnn_output_size = hidden_size

        self._dropout_p = dropout_p
        self.fc1 = torch.nn.Linear(rnn_output_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(p=self._dropout_p)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x_embedded = self.emb(x)
        # print(x_embedded.shape)

        _, (hidden, _) = self.rnn(x_embedded)
        hidden = hidden[-1, :, :]
        hidden = hidden.squeeze(0)

        out = self.fc1(hidden)
        out = self.dropout(out)
        out = self.relu(out)
        # print(out.shape)

        output = self.fc2(self.dropout(out))
        # print(output.shape)
        return output


def init_weights(m):
    for _, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)
