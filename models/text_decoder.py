import torch
import torch.nn as nn
import torch.nn.functional as functional


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


class AttnGRUTextDecoder(nn.Module):
    def __init__(
            self,
            output_size,
            hidden_size: int = 1024,
            embedding_size: int = 1024,
            dropout_proba: float = 0.1,
            max_length: int = 250
    ):
        super(AttnGRUTextDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout_proba = dropout_proba
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.embedding_projection = nn.Linear(self.embedding_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(
            self,
            tokenized_text: torch.LongTensor,
            hidden: torch.Tensor,
            encoder_outputs: torch.Tensor
    ):
        embedded = self.embedding(tokenized_text).view(1, 1, -1)
        embedded = self.dropout(embedded)
        embedded = self.embedding_projection(embedded)

        attention_input = torch.cat([embedded[0], hidden[0]], 1)

        attn_weights = functional.softmax(
            self.attn(attention_input), dim=1
        )

        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = functional.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights
