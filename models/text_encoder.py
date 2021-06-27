import torch
from torch import nn
from transformers import GPT2Model


class RUGPTTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str, eos_token_id: int, d_in: int, d_out: int):
        super(RUGPTTextEncoder, self).__init__()
        self.model = GPT2Model.from_pretrained(model_name)
        self.eos_token_id = eos_token_id
        self.projection = Projection(d_in, d_out)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.model(x, **kwargs)[0][(x == self.eos_token_id).nonzero(as_tuple=True)]
        x = self.projection(x)
        projection_len = torch.norm(x, dim=-1, keepdim=True)
        return x / projection_len


class Projection(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, p=0.5) -> None:
        super(Projection, self).__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    @classmethod
    def gelu(self, x):
        return x * torch.sigmoid(1.702 * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(self.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


if __name__ == '__main__':
    encoder = RUGPTTextEncoder(
        'sberbank-ai/rugpt3small_based_on_gpt2',
        eos_token_id=2,
        d_in=768,
        d_out=1024
    )

    print(encoder)
