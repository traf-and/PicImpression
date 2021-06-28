import torch
from torch.utils import data
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer  
import typing as tp
import pandas as pd

class Text_Dataset(Dataset):
    def __init__(
            self,
            tokenizer: GPT2Tokenizer,
            path: str,
            dataset_type: str = 'train',
    ):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(path)
        self.dataset_type = dataset_type

    def __getitem__(
        self,
        ind: int,
        max_seq_length: int =20,
        device: str = 'cpu'
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        texts = []
        tokenizer = self.tokenizer
        lines = [self.data[self.data.columns.tolist()[1]][ind]] # get impression str from index
        for text in lines:
            text = tokenizer(
                self.prc_text(text),
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
        input_ids = input_ids.to(device)
        attention_mask = torch.LongTensor([x["attention_mask"] for x in texts]).long().to(device)
        return input_ids, attention_mask

    @staticmethod
    def prc_text(text):
        text = text.replace("\n", " ").replace("\t", " ").replace("—", " ")
        return " ".join(text.split())