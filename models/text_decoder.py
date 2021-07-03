import torch
import torch.nn as nn


class GRUTextDecoder(nn.Module):
    def __init__(self, embedding_size: int = 256, hidden_size: int = 1024, output_size: int = 256):
        super(GRUTextDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.bos_token = output_size + 1
        self.embedding = nn.Embedding(output_size + 2, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):

        output = self.embedding(input)
        output = self.activation(output)
        output, hidden = self.gru(output.unsqueeze(0), hidden)
        output = self.out(output[0].unsqueeze(1))
        return output, hidden

