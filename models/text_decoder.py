import torch
import torch.nn as nn


class GRUTextDecoder(nn.Module):
    def __init__(self, hidden_size: int = 1024, output_size: int = 256):
        super(GRUTextDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        output = self.embedding(input).view(1, 1, -1)
        output = self.activation(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
