import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import random
from sklearn import preprocessing

class LRE17(Dataset):
    def __init__(self, hparams, data_type, is_load_wav=False):
        self.hparams = hparams
        self.df_full = pd.read_csv(self.hparams.wav_csv_path)
        self.data_type = data_type
        self.df = self.df_full[self.df_full['data_type'].str.contains(self.data_type)].reset_index(drop=True)
        list_language = self.df_full['lang'].unique()
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(list_language)
        self.is_load_wav = is_load_wav
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        lang = self.df.loc[idx, 'lang']
        if self.is_load_wav:
            wav_path = self.df.loc[idx, 'wav_path']
            wav, f = torchaudio.load(wav_path)
            if(wav.shape[0] != 1):
                wav = torch.mean(wav, dim=0)
                # wav = self.pad_crop_transform(wav)
        else:
            wav_id = self.df.loc[idx, 'utt_id']
            tensor_path = os.path.join(self.hparams.data_dir, self.hparams.upstream_model, self.data_type, wav_id + '.pt')
            wav = torch.load(tensor_path, map_location=torch.device('cpu'))
            wav = wav.squeeze(0)
            wav = wav.narrow(0, 0, wav.shape[0]//20*20)
        lang, = list(self.label_encoder.transform([lang]))
        return wav.view(-1, 20, 1024).detach(), lang
