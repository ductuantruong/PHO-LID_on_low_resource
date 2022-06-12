import os
import json

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

class LRE17Config(object):
    
    wav_csv_path = config['input']['wav_csv_path']
    
    run_name = config['input']['run_name']

    batch_size = int(config['optim_config']['batch_size'])
    
    epochs = int(config['optim_config']['epochs'])

    # LR of optimizer
    lr = float(config['optim_config']['learning_rate'])

    # No of GPUs for training and no of workers for datalaoders
    gpu = int(config['device'])
    
    n_workers = int(config['optim_config']['num_work'])

    # upstream model to be loaded from s3prl. Some of the upstream models are: wav2vec2, TERA, mockingjay etc.
    #See the available models here: https://github.com/s3prl/s3prl/blob/master/s3prl/upstream/README.md
    upstream_model = config['model_config']['upstream_model']

    model_type = config['model_config']['model_type']

    # feature dimension of upstream model. For example, 
    # For wav2vec2, feature_dim = 768
    feature_dim = config['model_config']['feat_dim']
    reduc_dim = config['model_config']['reduc_dim']
    d_k = config['model_config']['d_k']
    d_ff = config['model_config']['d_ff']
    n_heads = config['model_config']['n_heads']
    n_language = config['model_config']['n_language']

    # model checkpoint to continue from
    model_checkpoint = None
    

