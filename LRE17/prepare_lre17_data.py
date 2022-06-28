import os
from re import U
import shutil
import argparse
import scipy as sp
import pandas as pd

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--data_path',
                       metavar='data_path',
                       type=str,
                       default='/home/project/12001458/ductuan0/ISCAP_Age_Estimation/data/',
                       help='the path to dataset folder')
my_parser.add_argument('--metadata_path',
                       metavar='metadata_path',
                       type=str,
                       default='/home/project/12001458/ductuan0/ISCAP_Age_Estimation/data/',
                       help='the path to metadata folder')

args = my_parser.parse_args()


data_dir = args.data_path
metadata_dir = args.metadata_path


list_data_type = os.listdir(metadata_dir)

data_df = pd.DataFrame(columns=['utt_id', 'wav_path', 'data_type', 'lang'], index=None)
for data_type in list_data_type:
    lst_data_record = []
    print('Processing {} dataset'.format(data_type))
    with open(os.path.join(metadata_dir, data_type, 'utt2lang'), 'r') as utt2age_file:
        for line in utt2age_file.readlines():
            line = line.strip('\n')
            utt_id, lang = line.split(' ')
            wav_data_type = data_type if data_type == 'train' else 'test'
            data_record = {
                'utt_id': utt_id,
                'wav_path': os.path.join(data_dir, wav_data_type, utt_id+'.wav'),
                'data_type': data_type,
                'lang': lang,
            }
            lst_data_record.append(data_record)
    data_df = pd.concat([data_df, pd.DataFrame.from_records(lst_data_record)], ignore_index=True)
data_df.to_csv(os.path.join(data_dir, 'data_info_age.csv'), index=False)
print('Data saved at ', data_dir)