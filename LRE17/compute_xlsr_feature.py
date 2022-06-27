import torch
import torchaudio
import librosa

import os
import argparse
from tqdm import tqdm

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--device',
                       metavar='device',
                       type=str,
                       default='cuda',
                       help='the type of computing device')
my_parser.add_argument('--data_path',
                       metavar='data_path',
                       type=str,
                       default='/home/project/12001458/LRE22/LRE17/JC_data/data/wav',
                       help='the path to dataset folder')
my_parser.add_argument('--wav_path',
                       metavar='wav_path',
                       type=str,
                       default='train',
                       help='the path to wav folder')

args = my_parser.parse_args()

device = args.device
original_data_dir = args.data_path
xlsr_folder_path = os.path.join(original_data_dir, 'xlsr')

wav_dir = args.wav_path

upstream_model = torch.hub.load('s3prl/s3prl', 'wav2vec2_xlsr').to(device)
upstream_model_cpu = torch.hub.load('s3prl/s3prl', 'wav2vec2_xlsr').to('cpu')

print("Processing {} ...".format(wav_dir))
xlsr_lang_path = os.path.join(xlsr_folder_path, wav_dir)
os.makedirs(xlsr_lang_path, exist_ok=True)
    
wav_folder_path = os.path.join(original_data_dir, xlsr_folder_path)
for wav_file in tqdm(os.listdir(wav_folder_path)):
    wav_file_name = wav_file[:-4]
    save_path = os.path.join(xlsr_lang_path, '{}.pt'.format(wav_file_name))
    if os.path.exists(save_path):
        continue
    wav, f = librosa.load(os.path.join(wav_folder_path, wav_file))
    wav = torch.from_numpy(librosa.util.normalize(wav))
    wav = librosa.effects.preemphasis(wav)
    wav = wav.to(device)
    try:
        wav = wav.to(device)
        feature = upstream_model(wav)['last_hidden_state']
    except:
        try:
            wav = wav.cpu()
            feature = upstream_model_cpu(wav)['last_hidden_state']
        except:
            with open('error_wav.txt', 'a') as f:
                f.writelines(os.path.join(wav_folder_path, wav_file) +  ' ' + str(wav.shape)  + '\n')
            del wav
            continue
    torch.save(feature, save_path)
    del feature
        
