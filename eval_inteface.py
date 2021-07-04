import torch
import torch.nn as nn
import typing as tp

from models.Image_encoder import VisualTransformer
from models.text_decoder import GRUTextDecoder
from models.text_encoder import RUGPTTextEncoder
import numpy as np

class PicAnnotatorEvalInteface(object):
    def __init__(
            self,
            decoder: GRUTextDecoder,
            img_encoder: tp.Optional[VisualTransformer] = None,
            txt_encoder: tp.Optional[RUGPTTextEncoder] = None,
            device: str = 'cpu'
    ):
        self.decoder = decoder.to(device)
        self.img_encoder = img_encoder.to(device)
        self.txt_encoder = txt_encoder.to(device)
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
        return text

    def img_to_text(self, img: np.array):
        # TODO
        return None
