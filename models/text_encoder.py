import torch
import typing as tp
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer


def prc_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def get_text_batch(
        lines: tp.List[str],
        tokenizer: GPT2Tokenizer,
        max_seq_length: int,
        use_cpu: bool = False
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    texts = []
    for text in lines:
        text = tokenizer(
            prc_text(text),
            padding="max_length",
            max_length=max_seq_length,
            truncation=True
        )
        text["input_ids"] = [tokenizer.bos_token_id] + text["input_ids"]
        text["attention_mask"] = [1] + text["attention_mask"]
        text["attention_mask"] = text["attention_mask"][:max_seq_length]
        text["input_ids"] = text["input_ids"][:max_seq_length]
        try:
            pos = text["input_ids"].index(tokenizer.pad_token_id)
        except ValueError:
            pos = -1
        text["input_ids"][pos] = tokenizer.eos_token_id
        text["attention_mask"][pos] = 1
        texts.append(text)
    input_ids = torch.LongTensor([x["input_ids"] for x in texts]).long()
    device = input_ids.device if use_cpu else torch.cuda.current_device()
    input_ids = input_ids.to(device)
    attention_mask = torch.LongTensor([x["attention_mask"] for x in texts]).long().to(device)
    return input_ids, attention_mask


class RUGPTTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str, eos_token_id: int, d_in: int, d_out: int):
        super(RUGPTTextEncoder, self).__init__()
        self.model = GPT2Model.from_pretrained(model_name)
        self.tokenizer = self._initialize_tokenizer(model_name)
        self.eos_token_id = eos_token_id
        self.projection = Projection(d_in, d_out)

    @staticmethod
    def _initialize_tokenizer(model_name: str):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        add_tokens = tokenizer.add_special_tokens({"bos_token": "<s>"})
        assert add_tokens == 0
        add_tokens = tokenizer.add_special_tokens({"eos_token": "</s>"})
        assert add_tokens == 0
        add_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert add_tokens == 0
        return tokenizer

    def tokenize_batch(
            self,
            texts: tp.List[str],
            max_seq_length: int = 20
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return get_text_batch(
            texts,
            self.tokenizer,
            max_seq_length=max_seq_length,
            use_cpu=True
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        x = self.model(x, attention_mask=attention_mask)[0][(x == self.eos_token_id).nonzero(as_tuple=True)]
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
    ).cpu()
    sd = torch.load('text_encoder.ckpt')
    encoder.load_state_dict(sd)

    tokens, att_mask = encoder.tokenize_batch(['кек'])
    print(encoder(tokens, att_mask)[0])
