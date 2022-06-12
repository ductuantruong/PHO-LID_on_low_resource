import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torchaudio
import random
from sklearn import preprocessing

class LRE17(data.Dataset):
    def __init__(self, data_type, hparams):
        self.df_full = pd.read_csv(hparams.wav_csv_path)
        self.df = self.df_full[self.df_full['data_type'] == data_type].reset_index(drop=True)
        list_language = self.df_full['lang'].unique()
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(list_language)
        
    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        lang = self.df.loc[idx, 'lang']
        wav_path = self.df.loc[idx, 'wav_path']

        wav, f = torchaudio.load(wav_path)
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
            wav = self.resampleUp(wav)
            # wav = self.pad_crop_transform(wav)

        return wav, self.label_encoder.transform(lang)
    
def get_atten_mask(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()

def get_atten_mask_frame(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len*20, max_len*20])
    for i in range(batch_size):
        length = seq_lens[i]*20
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()


def std_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]
    weight_unbaised = torch.tensor(seq_lens) / (torch.tensor(seq_lens) - 1)
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length = seq_lens[i]
        atten_mask[i, length:, :] = 1e-9
    return atten_mask, weight_unbaised

def mean_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]
    weight_unbaised = seq_lens[0] / torch.tensor(seq_lens)
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length = seq_lens[i]
        atten_mask[i, :length, :] = 0
    return atten_mask.bool(), weight_unbaised