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
        index: int,
        max_seq_length: int = 20,
        device: str = 'cpu'
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        texts = []
        tokenizer = self.tokenizer
        lines = [self.data[self.data.columns.tolist()[1]][index]] # get impression str from index
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
        input_ids = input_ids[0,:]
        attention_mask = attention_mask[0,:]
        return input_ids, attention_mask

    def collate_fn(self, data):
        batch_size = len(data)
        input_ids, att_mask = list(zip(*data))
        _max_length_ids = max([i.shape[0] for i in input_ids])
        _max_length_att = max([i.shape[0] for i in att_mask])
        input_ids_batch = torch.zeros((batch_size, _max_length_ids), dtype=torch.long)
        attn_mask_batch = torch.zeros((batch_size, _max_length_att), dtype=torch.long)
        for i, (input_ids, att_mask) in enumerate(data):
            input_ids = torch.nn.functional.pad(input_ids, (0, _max_length_ids - input_ids.shape[0]))
            att_mask = torch.nn.functional.pad(att_mask, (0, _max_length_att - att_mask.shape[0]))
            input_ids_batch[i, :] = input_ids
            attn_mask_batch[i, :] = att_mask
        return input_ids_batch, attn_mask_batch


    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def prc_text(text:str):
        text = text.replace("\n", " ").replace("\t", " ").replace("—", " ").replace("'", " ").replace('"', " ").replace("-", " ")
        return " ".join(text.split())