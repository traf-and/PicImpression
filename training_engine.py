from os import path
import torch
import typing as tp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch.optim as optim
from torch.utils.data import dataset, DataLoader
from transformers import GPT2Tokenizer
from transformers.utils.dummy_pt_objects import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING 
from models.text_decoder import GRUTextDecoder, AttnGRUTextDecoder
from models.text_encoder import RUGPTTextEncoder
from models.dataset_create.text_data import Text_Dataset

class LightningEngine(pl.LightningModule):
    def __init__(
            self,
            text_encoder: RUGPTTextEncoder,
            text_decoder: GRUTextDecoder,
            criterion: torch.nn.Module,
            tokenizer: GPT2Tokenizer,
            optimizer: optim
    ):
        super(LightningEngine, self).__init__()
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        tokenized_text, att_mask = batch
        with torch.no_grad():
            enc_out = self.text_encoder(tokenized_text, att_mask).unsqueeze(0)  # B, 1024
        loss = torch.Tensor([0.0]).to(next(self.text_decoder.parameters()).device)
        decoder_input = torch.LongTensor([[self.tokenizer.bos_token_id] for _ in range(tokenized_text.shape[0])])
        decoder_input = decoder_input.to(next(self.text_decoder.parameters()).device)
        decoder_hidden = self.text_decoder.init_hidden(tokenized_text.shape[0])
        for i in range(tokenized_text.shape[1]):
            to_predict = tokenized_text[:, i]
            output, decoder_hidden = self.text_decoder(decoder_input, decoder_hidden, enc_out)
            loss += self.criterion(output.squeeze(1), to_predict)
            decoder_input = to_predict
            # decoder_hidden = decoder_hidden.detach()

        loss = loss / tokenized_text.shape[1]

        self.log('train/loss', loss.cpu().item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        tokenized_text, att_mask = batch
        enc_out = self.text_encoder(tokenized_text, att_mask).unsqueeze(0)  # B, 1024
        loss = torch.Tensor([0.0]).to(next(self.text_decoder.parameters()).device)
        decoder_input = torch.LongTensor([[self.tokenizer.bos_token_id] for _ in range(tokenized_text.shape[0])])
        decoder_input = decoder_input.to(next(self.text_decoder.parameters()).device)
        decoder_hidden = self.text_decoder.init_hidden(tokenized_text.shape[0])
        for i in range(tokenized_text.shape[1]):
            to_predict = tokenized_text[:, i]
            output, decoder_hidden = self.text_decoder(decoder_input, decoder_hidden, enc_out)
            loss += self.criterion(output.squeeze(1), to_predict)
            decoder_input = to_predict
            # decoder_hidden = decoder_hidden.detach()

        loss = loss / tokenized_text.shape[1]

        self.log('valid/loss', loss.cpu().item())
        return loss

    def configure_optimizers(self):
        return self.optimizer

import sys

if __name__ == '__main__':
    #for param in sys.argv:
    #    model_state = param
    model_state = 'train'
    encoder = RUGPTTextEncoder(
        'sberbank-ai/rugpt3small_based_on_gpt2',
        eos_token_id=2,
        d_in=768,
        d_out=1024
    ).cpu()
    sd = torch.load('/home/artem/proj/PictureToText/In_work/text_encoder.ckpt')
    encoder.load_state_dict(sd)
    path = 'Impressive Dataset/20k_vpechatlator_29_02_20.csv'
    val_path = '/home/artem/proj/PictureToText/Pretrain_DS/poem.csv'
    data_set = Text_Dataset(encoder.tokenizer, path)
    data = DataLoader(data_set, batch_size=400, num_workers=8, pin_memory=True, collate_fn=data_set.collate_fn)
    val_data_set = Text_Dataset(encoder.tokenizer, val_path)
    val_dataloaders = DataLoader(val_data_set, batch_size=400, num_workers=8, pin_memory=True, collate_fn=data_set.collate_fn)
    
    decoder = GRUTextDecoder(
        hidden_size=1024,
        embedding_size=1024,
        encoder_output_size=1024,
        output_size=len(encoder.tokenizer)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    model = LightningEngine(encoder, decoder, criterion=criterion, tokenizer=encoder.tokenizer, optimizer=optimizer)
    #model = LightningEngine.load_from_checkpoint('pl_ckpt/epoch=21-step=73391.ckpt', text_encoder=encoder, text_decoder=decoder, criterion=criterion, optimizer=optimizer)
    checkpoint_callback = ModelCheckpoint(dirpath='pl_ckpt/')
    logger = TensorBoardLogger("tb_logs/", name="my_model")
    trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback], logger=logger)
    if model_state == 'train':
        trainer.fit(model, data)
    elif model_state == 'valid':
        val_loss=trainer.validate(model, val_dataloaders=val_dataloaders)