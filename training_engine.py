from os import path
import torch
import typing as tp
import pytorch_lightning as pl
from torch import nn
import torch.optim as optim
from torch.utils.data import dataset, DataLoader
from transformers import GPT2Tokenizer 
from models.text_decoder import GRUTextDecoder
from models.text_encoder import RUGPTTextEncoder
from models.dataset_create.text_data import Text_Dataset

class LightningEngine(pl.LightningModule):
    def __init__(
            self,
            text_encoder: RUGPTTextEncoder,
            text_decoder: GRUTextDecoder,
            criterion: torch.nn.Module,
            optimizer: optim
    ):
        super(LightningEngine, self).__init__()
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.criterion = criterion

        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        tokenized_text, att_mask = batch

        hidden = self.text_encoder(tokenized_text, att_mask).unsqueeze(0)  # B, 1024
        loss = torch.Tensor([0.0]).to(next(self.model.parameters()).device)
        decoder_input = torch.LongTensor([self.text_decoder.bos_token for _ in range(tokenized_text.shape[0])])
        decoder_input = decoder_input.to(next(self.model.parameters()).device)
        for i in range(tokenized_text.shape[1]):
            to_predict = tokenized_text[:, i]
            output, hidden = self.text_decoder(decoder_input, hidden)
            loss += self.criterion(output.squeeze(1), to_predict)
            decoder_input = to_predict
            hidden = hidden.detach()

        self.log('train/loss', loss.cpu().item())
        return loss

    def configure_optimizers(self):
        return self.optimizer


if __name__ == '__main__':
    encoder = RUGPTTextEncoder(
        'sberbank-ai/rugpt3small_based_on_gpt2',
        eos_token_id=2,
        d_in=768,
        d_out=1024
    ).cpu()
    sd = torch.load('/home/artem/proj/PictureToText/In_work/text_encoder.ckpt')
    encoder.load_state_dict(sd)
    
    path = '/home/artem/proj/PictureToText/Pretrain_DS/prose.csv'
    data_set = Text_Dataset(encoder.tokenizer, path)
    data = DataLoader(data_set, batch_size=1, num_workers=8, pin_memory=True, collate_fn=data_set.collate_fn)
    decoder = GRUTextDecoder(output_size=len(encoder.tokenizer))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    model = LightningEngine(encoder, decoder, criterion=criterion, optimizer=optimizer)
    trainer = pl.Trainer(gpus=0)
    trainer.fit(model, data)