import os
from re import U
from pydub.utils import mediainfo
import argparse
import scipy as sp
import pandas as pd
from tqdm.contrib.concurrent import process_map

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--data_path',
                       metavar='data_path',
                       type=str,
                       default='/home/project/12001458/LRE22/LRE17/JC_data/data/wav',
                       help='the path to dataset folder')
my_parser.add_argument('--metadata_path',
                       metavar='metadata_path',
                       type=str,
                       default='/home/project/12001458/LRE22/LRE17/JC_data/metadata/',
                       help='the path to metadata folder')

args = my_parser.parse_args()


def get_data_info(line):
    line = line.strip('\n')
    utt_id, lang = line.split(' ')
    wav_data_type = data_type if data_type == 'train' else 'eval'
    wav_path = os.path.join(data_dir, wav_data_type, utt_id+'.wav')
    if not os.path.exists(wav_path):
        return dict()
    duration = mediainfo(wav_path)['duration']
    data_record = {
        'utt_id': utt_id,
        'wav_path': os.path.join(data_dir, wav_data_type, utt_id+'.wav'),
        'duration': duration,
        'data_type': data_type,
        'lang': lang,
    }
    return data_record

data_dir = args.data_path
metadata_dir = args.metadata_path


list_data_type = os.listdir(metadata_dir)

data_df = pd.DataFrame(columns=['utt_id', 'wav_path', 'duration', 'data_type', 'lang'], index=None)
for data_type in list_data_type:
    lst_data_record = []
    print('Processing {} dataset'.format(data_type))
    with open(os.path.join(metadata_dir, data_type, 'utt2lang'), 'r') as utt2age_file:
        metainfo = process_map(get_data_info,
                                utt2age_file.readlines(),
                                max_workers = 20,
                                chunksize = 15)
    data_df = pd.concat([data_df, pd.DataFrame.from_records(metainfo)], ignore_index=True)
data_df.dropna(inplace=True)
data_df.to_csv(os.path.join(data_dir, 'data_info_age.csv'), index=False)
print('Data saved at ', data_dir)
