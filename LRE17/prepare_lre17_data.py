import os
import argparse
from tokenize import group
import pandas as pd
from os.path import isfile

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--path',
                       metavar='path',
                       type=str,
                       default='data/wav',
                       help='the path to dataset folder')
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


original_data_dir = args.path
train_folder_name = os.path.join(args.train_folder_name)
val_folder_name = os.path.join(args.val_folder_name)

train_data_path = os.path.join(original_data_dir, train_folder_name)
val_data_path = os.path.join(original_data_dir, val_folder_name)

list_lang_folder = []
for folder in os.listdir(train_data_path):
    if os.path.isdir(os.path.join(train_data_path, folder)):
        list_lang_folder.append(folder)

dict_data_type = {
                    'TRAIN': train_data_path, 
                    'VAL': val_data_path
                }

data_df = pd.DataFrame(columns=['utt_id', 'data_type', 'wav_path', 'group_lang', 'lang'], index=None)

list_data_record = []
for data_type in dict_data_type.keys():
    print('Processing {} dataset'.format(data_type))
    data_path = dict_data_type[data_type]
    for lang_name in list_lang_folder:
        group_lang = lang_name.split('_')[0]
        lang = lang_name.split('_')[1]
        for wav_file in os.listdir(os.path.join(data_path, lang_name)):
            data_record = {
                'utt_id': wav_file[:-4],
                'data_type': data_type,
                'wav_path': os.path.join(data_path, wav_file),
                'group_lang': group_lang,
                'lang': lang
            }
            list_data_record.append(data_record)
data_df = pd.concat([data_df, pd.DataFrame.from_records(list_data_record)], ignore_index=True)
data_df.to_csv(os.path.join(original_data_dir, 'data_info.csv'), index=False)
print('Data saved at ', original_data_dir)
