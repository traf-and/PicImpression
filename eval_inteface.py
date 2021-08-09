from tokenizers import Tokenizer
import torch
from torch._C import device
import torch.nn as nn
import typing as tp

from models.Image_encoder import VisualTransformer
from models.text_decoder import GRUTextDecoder
from models.text_encoder import RUGPTTextEncoder
from transformers import GPT2Tokenizer 
import numpy as np

class PicAnnotatorEvalInteface(object):
    def __init__(
            self,
            decoder: GRUTextDecoder,
            img_encoder: tp.Optional[VisualTransformer] = None,
            txt_encoder: tp.Optional[RUGPTTextEncoder] = None,
            tokenizer: tp.Optional[GPT2Tokenizer] = None,
            device: str = 'cpu'
    ):
        self.decoder = decoder.to(device)
        self.img_encoder = img_encoder.to(device)
        self.txt_encoder = txt_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_img_encoder(
            cls,
            decoder_path: str,
            img_encoder_path: str,
            device: str = 'cpu'
    ):
        checkpoint = torch.load(decoder_path)
        decoder = GRUTextDecoder(**checkpoint['params'])
        decoder.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(img_encoder_path)
        img_encoder = VisualTransformer(**checkpoint['params'])
        img_encoder.load_state_dict(checkpoint['state_dict'])

        return cls(decoder=decoder, img_encoder=img_encoder, device=device)

    @classmethod
    def from_txt_encoder(
            cls,
            decoder_path: str,
            txt_encoder_path: str,
            device: str = 'cpu'
    ):
        checkpoint = torch.load(decoder_path)
        decoder = GRUTextDecoder(**checkpoint['params'])
        decoder.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(txt_encoder_path)
        txt_encoder = RUGPTTextEncoder(**checkpoint['params'])
        txt_encoder.load_state_dict(checkpoint['state_dict'])

        return cls(decoder=decoder, txt_encoder=txt_encoder, device=device)

    def text_to_text(self, text: str):
        # TODO
        decoder_path = ''
        txt_encoder_path = ''
        token = self.tokenize_text(text)
        tt_model = self.from_txt_encoder(decoder_path=decoder_path, 
            txt_encoder_path=txt_encoder_path,
            device=self.device)
        mod_out = tt_model(token)
        text = self.tokenizer.batch_decode(mod_out)
        return text

    def img_to_text(self, img: np.array):
        # TODO
        decoder_path = ''
        image_encoder_path = ''
        it_model = self.from_img_encoder(decoder_path=decoder_path,
            img_encoder_path=image_encoder_path,
            device=self.device)
        mod_out = it_model(img)
        text = self.tokenizer.batch_decode(mod_out)
        return text

    def tokenize_text(self, lines: str):
        tokenizer = self.tokenizer
        max_seq_length = 20
        texts = []
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
        input_ids = input_ids.to(self.device)
        attention_mask = torch.LongTensor([x["attention_mask"] for x in texts]).long().to(self.device)
        return input_ids, attention_mask

    @staticmethod
    def prc_text(text:str):
        text = text.replace("\n", " ").replace("\t", " ").replace("—", " ").replace("'", " ").replace('"', " ").replace("-", " ")
        return " ".join(text.split())

if __name__ == '__main__':
    text_encoder = RUGPTTextEncoder(
        'sberbank-ai/rugpt3small_based_on_gpt2',
        eos_token_id=2,
        d_in=768,
        d_out=1024
    ).cpu()
    te_sd = torch.load('In_work/text_encoder.ckpt')
    text_encoder.load_state_dict(te_sd)
    tokenizer = text_encoder.tokenizer

    image_encoder = VisualTransformer()
    ie_sd = torch.load('In_work/visual_encoder.ckpt')
    image_encoder.load_state_dict(ie_sd)

    d_sd = {}
    ckpt = (torch.load('pl_ckpt/epoch=21-step=73391.ckpt')['state_dict'])
    for i in ckpt.keys():
        if i[5:12] == 'decoder':
            d_sd[i] = ckpt[i]
    decoder = GRUTextDecoder(output_size=len(tokenizer))
    decoder.load_state_dict(d_sd)

    dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PicAnnotatorEvalInteface(decoder = decoder,
            img_encoder = image_encoder,
            txt_encoder = text_encoder,
            tokenizer = tokenizer,
            device = device).eval()
    