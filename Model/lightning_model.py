import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy

import math
import pandas as pd
import torch_optimizer as optim

from Model.models import CNN_Trans_LID, PHOLID
from utils.ssl_sampler import Phoneme_SSL_loss

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.models = {
            'PHOLID': PHOLID,
        }
        
        self.model = self.models[HPARAMS['model_type']](
                                                        input_dim=HPARAMS['input_dim'], 
                                                        feat_dim=HPARAMS['feat_dim'],
                                                        d_k=HPARAMS['feat_dim'],
                                                        d_v=HPARAMS['feat_dim'],
                                                        n_heads=HPARAMS['n_heads'], 
                                                        d_ff=HPARAMS['d_ff'] 
                                                    )
        self.weight_lid = HPARAMS["weight_lid"]
        self.weight_ssl = HPARAMS["weight_ssl"]

        self.accuracy = Accuracy()
        self.loss_func_phn = Phoneme_SSL_loss(num_frames=20, num_sample=HPARAMS["nega_frames"])
        self.loss_func_lid = nn.CrossEntropyLoss()

        self.lr = HPARAMS['lr']
        self.SSL_epochs = HPARAMS['SSL_epochs']
        self.warmup = HPARAMS['warmup']
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_len):
        return self.model(x, x_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        def lr_foo(epoch):
            if epoch <= self.SSL_epochs:
                lr_scale = 1
            elif epoch < self.SSL_epochs + self.warmup:
                lr_scale = (epoch - self.SSL_epochs) / self.warmup
            else:
                lr_scale = 0.5 * (math.cos((epoch - self.SSL_epochs - self.warmup) / (self.max_epochs - self.SSL_epochs - self.warmup) * math.pi) + 1)
            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y_lang, x_len = batch
             
        y_lang_hat, y_pho = self(x, x_len, is_train=True)

        if self.current_epoch < self.SSL_epochs:
            loss_phn = self.loss_func_phn(y_pho, x_len)
            loss_lid = self.loss_func_lid(y_lang_hat, y_lang)
            loss = loss_phn
        else:
            loss_phn = self.loss_func_phn(y_pho, x_len)
            loss_lid = self.loss_func_lid(y_lang_hat, y_lang)
            loss = self.weight_lid*loss_lid + self.weight_ssl*loss_phn

        acc = self.accuracy(y_lang_hat, y_lang)
        return {
                'loss': loss, 
                'acc':acc
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
        if self.current_epoch < self.SSL_epochs:
            loss_phn = self.loss_func_phn(y_pho, x_len)
            loss_lid = self.loss_func_lid(y_lang_hat, y_lang)
            loss = loss_phn
        else:
            loss_phn = self.loss_func_phn(y_pho, x_len)
            loss_lid = self.loss_func_lid(y_lang_hat, y_lang)
            loss = self.weight_lid*loss_lid + self.weight_ssl*loss_phn

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
                'acc':acc
            }

    def test_epoch_end(self, outputs):
        acc = torch.tensor([x['test_acc'] for x in outputs]).mean()

        pbar = {
                'test_acc':acc.item()
            }

        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
