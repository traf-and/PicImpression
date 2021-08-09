import torch
import torch.nn as nn
import torch.nn.functional as functional


class GRUTextDecoder(nn.Module):
    def __init__(
            self,
            output_size: int,
            embedding_size: int = 1024,
            hidden_size: int = 1024,
            encoder_output_size: int = 1024,
    ):
        super(GRUTextDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation = nn.LeakyReLU(0.2)

        self.encoder_decoder_adaptor = nn.Linear(embedding_size + encoder_output_size, embedding_size)

    def forward(
            self,
            input: torch.Tensor,
            hidden: torch.Tensor,
            encoder_output: torch.Tensor
    ):
        embedded = self.embedding(input)

        attention_input = torch.cat([embedded, encoder_output], 2)

        embedded = functional.softmax(
            self.encoder_decoder_adaptor(attention_input), dim=1
        )

        embedded = self.activation(embedded)
        embedded, hidden = self.gru(embedded.transpose(0, 1), hidden)
        embedded = self.out(embedded[0].unsqueeze(1))
        return embedded, hidden

    def init_hidden(self, batch_size: int = 1):
        return torch.zeros(1, batch_size, self.hidden_size, device=next(self.parameters()).device)


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
        self.dropout = nn.Dropout(self.dropout_proba)
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

        attention_input = torch.cat([embedded[0], hidden[0]], 2)

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


if __name__ == '__main__':
    model = GRUTextDecoder(output_size=1024)
    hidden = model.init_hidden(2)
    e_out = torch.randn((2, 1, 1024))
    tokenized_text = torch.LongTensor([[0], [3]])

    d_out, hidden = model(tokenized_text, hidden, e_out)
    d_out, hidden = model(tokenized_text, hidden, e_out)
