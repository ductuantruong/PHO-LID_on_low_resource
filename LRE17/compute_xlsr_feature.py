import torch
import torchaudio

import os
import argparse
from tqdm import tqdm

from zmq import device

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--device',
                       metavar='device',
                       type=str,
                       default='cuda',
                       help='the type of computing device')
my_parser.add_argument('--data_path',
                       metavar='data_path',
                       type=str,
                       default='data',
                       help='the path to dataset folder')
my_parser.add_argument('--wav_path',
                       metavar='wav_path',
                       type=str,
                       default='data/wav',
                       help='the path to wav folder')
my_parser.add_argument('--train_folder_name',
                       metavar='train_folder_name',
                       type=str,
                       default='lre-17-train',
                       help='the name of train data folder')
my_parser.add_argument('--val_folder_name',
                       metavar='val_folder_name',
                       type=str,
                       default='lre-17-eval',
                       help='the name of validate data folder')

args = my_parser.parse_args()

device = args.device
original_data_dir = args.data_path
xlsr_folder_path = os.path.join(original_data_dir, 'xlsr')

original_wav_dir = args.wav_path
train_folder_name = os.path.join(args.train_folder_name)
val_folder_name = os.path.join(args.val_folder_name)

train_data_path = os.path.join(original_wav_dir, train_folder_name)
val_data_path = os.path.join(original_wav_dir, val_folder_name)

upstream_model = torch.hub.load('s3prl/s3prl', 'wav2vec2_xlsr').to(device)

list_data_path = [train_data_path, val_data_path]
for data_type_path in list_data_path:
    for folder in os.listdir(data_type_path):
        xlsr_lang_path = os.path.join(xlsr_folder_path, data_type_path, folder)
        if os.path.exists(xlsr_lang_path):
            os.mkdir(xlsr_lang_path)
        else:
            break
        
        wav_folder_path = os.path.isdir(os.path.join(data_type_path, folder))
        for wav_file in tqdm(os.listdir(wav_folder_path)):
            wav_file_name = wav_file[:-4]
            wav, f = torchaudio.load(os.path.join(wav_folder_path, wav_file)).to(device)
            feature = upstream_model(wav)['last_hidden_state']
            torch.save(feature, os.path.join(xlsr_lang_path, '{}.pt'.format(wav_file_name)))
            



