import torch
import typing as tp
import pytorch_lightning as pl

from models.text_decoder import GRUTextDecoder
from models.text_encoder import RUGPTTextEncoder


class LightningEngine(pl.LightningModule):
    def __init__(
            self,
            text_encoder: RUGPTTextEncoder,
            text_decoder: GRUTextDecoder,
            criterion: torch.nn.Module,
            optimizer: torch.optim.optimizer.Optimizer
    ):
        super(LightningEngine, self).__init__()
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.criterion = criterion

        self.optimizer = optimizer
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        tokenized_text, att_mask = batch
        # B, SeqLength // B, 123
        hidden = self.text_encoder(tokenized_text, att_mask)  # B, 1024

        loss = torch.Tensor([0.0])
        decoder_input = torch.Tensor([[self.text_decoder.bos_token]]).to(self.text_decoder.device)
        for i in range(tokenized_text.shape[1]):
            to_predict = tokenized_text[:, i]
            output, hidden = self.text_decoder(decoder_input, hidden)
            loss += self.criterion(output, to_predict)
            decoder_input = to_predict

        self.log('train/loss', loss.cpu().item())
        return loss

    def configure_optimizers(self):
        return self.optimizer
