import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy

import pandas as pd
import torch_optimizer as optim

from Model.models import CNN_Trans_LID, PHOLID

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.models = {
            'PHOLID': PHOLID,
        }
        
        self.model = self.models[HPARAMS['model_type']](
                                                        upstream_model=HPARAMS['upstream_model'], 
                                                        input_dim=HPARAMS['input_dim'], 
                                                        feat_dim=HPARAMS['feat_dim'],
                                                        d_k=HPARAMS['feat_dim'],
                                                        d_v=HPARAMS['feat_dim'],
                                                        n_heads=HPARAMS['n_heads'], 
                                                        d_ff=HPARAMS['d_ff'] 
                                                    )
        
        self.accuracy = Accuracy()
        self.loss_module = nn.CrossEntropyLoss()

        self.lr = HPARAMS['lr']

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_len):
        return self.model(x, x_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y_lang, x_len = batch
        
        y_lang_hat, y_pho = self(x, x_len)

        loss = self.loss_module(y_lang_hat, y_lang)
        acc = self.accuracy(y_lang_hat, y_lang)

        return {
                'loss': loss, 
                'accuracy': acc,
            }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        acc = torch.tensor([x['acc'] for x in outputs]).mean()

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc',acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_lang, x_len = batch
        
        y_lang_hat, y_pho = self(x, x_len)

        loss = self.loss_module(y_lang_hat, y_lang)
        acc = self.accuracy(y_lang_hat, y_lang)

        return {
                'val_loss':loss, 
                'val_acc':acc
            }

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        acc = torch.tensor([x['val_acc'] for x in outputs]).mean()
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_lang, x_len = batch
        y_lang_hat, y_pho = self(x, x_len)

        acc = self.accuracy(y_lang_hat, y_lang)

        return {
                'test_acc':acc
            }

    def test_epoch_end(self, outputs):
        acc = torch.tensor([x['test_acc'] for x in outputs]).mean()

        pbar = {
                'test_acc':acc.item()
            }

        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
