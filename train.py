from config import LRE17Config
from argparse import ArgumentParser
from multiprocessing import Pool
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer

import torch
import torch.utils.data as data
import random
import numpy as np

# SEED
def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pl.utilities.seed.seed_everything(seed)

seed_torch()

from LRE17.data_load import LRE17
from Model.lightning_model import LightningModel

import torch.nn.utils.rnn as rnn_utils

# def collate_fn(batch):
    # batch.sort(key=lambda x: len(x[1]), reverse=True)
    # seq, label = zip(*batch)
    # seq_length = [len(x) for x in label]
    # data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    # label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0)
    # return data, label, seq_length

def collate_fn_atten(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seq, labels, seq_length = zip(*batch)
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return data, labels, seq_length

if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--wav_csv_path', type=str, default=LRE17Config.wav_csv_path)
    parser.add_argument('--batch_size', type=int, default=LRE17Config.batch_size)
    parser.add_argument('--epochs', type=int, default=LRE17Config.epochs)
    parser.add_argument('--num_layers', type=int, default=LRE17Config.num_layers)
    parser.add_argument('--feature_dim', type=int, default=LRE17Config.feature_dim)
    parser.add_argument('--lr', type=float, default=LRE17Config.lr)
    parser.add_argument('--gpu', type=int, default=LRE17Config.gpu)
    parser.add_argument('--n_workers', type=int, default=LRE17Config.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=LRE17Config.model_checkpoint)
    parser.add_argument('--run_name', type=str, default=LRE17Config.run_name)
    parser.add_argument('--model_type', type=str, default=LRE17Config.model_type)
    parser.add_argument('--upstream_model', type=str, default=LRE17Config.upstream_model)
    parser.add_argument('--narrow_band', type=str, default=LRE17Config.narrow_band)
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Check device
    if not torch.cuda.is_available():
        hparams.gpu = 0
    else:        
        print(f'Training Model on LRE17 Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = LRE17(
        data_type = 'TRAIN',
        hparams = hparams
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=hparams.batch_size, 
        shuffle=True, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn_atten,
    )
    ## Validation Dataset
    valid_set = LRE17(
        data_type = 'VAL',
        hparams = hparams,
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=1,
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn_atten,
    )
    ## Testing Dataset
    test_set = LRE17(
        hparams = hparams,
        is_train=False
    )
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=1,
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn_atten,
    )

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))

    logger = WandbLogger(
        name=LRE17Config.run_name,
        offline=True,
        project='LID'
    )

    model = LightningModel(vars(hparams))

    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/{}'.format(hparams.run_name),
        monitor='val/loss', 
        mode='min',
        verbose=1)

    trainer = Trainer(
        fast_dev_run=hparams.dev, 
        gpus=hparams.gpu, 
        max_epochs=hparams.epochs, 
        checkpoint_callback=True,
        callbacks=[
            EarlyStopping(
                monitor='val/loss',
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode='min'
                ),
            model_checkpoint_callback
        ],
        logger=logger,
        resume_from_checkpoint=hparams.model_checkpoint,
        distributed_backend='ddp',
        auto_lr_find=True
        )

    # Fit model
    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    print('\n\nCompleted Training...\nTesting the model with checkpoint -', model_checkpoint_callback.best_model_path)
