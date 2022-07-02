import os
import json

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

class LRE17Config(object):
    
    wav_csv_path = config['input']['wav_csv_path']
    
    data_dir = config['input']['data_dir']
    
    run_name = config['input']['run_name']

    batch_size = int(config['optim_config']['batch_size'])
    
    epochs = int(config['optim_config']['epochs'])
    SSL_epochs = int(config['optim_config']['SSL_epochs'])
    # LR of optimizer
    lr = float(config['optim_config']['learning_rate'])
    weight_lid = float(config['optim_config']['weight_lid'])
    weight_ssl = float(config['optim_config']['weight_ssl'])
    
    # No of GPUs for training and no of workers for datalaoders
    gpu = int(config['optim_config']['gpu'])
    
    n_workers = int(config['optim_config']['num_work'])

    # upstream model to be loaded from s3prl. Some of the upstream models are: wav2vec2, TERA, mockingjay etc.
    #See the available models here: https://github.com/s3prl/s3prl/blob/master/s3prl/upstream/README.md
    upstream_model = config['model_config']['upstream_model']

    model_type = config['model_config']['model_type']

    # feature dimension of upstream model. For example, 
    # For wav2vec2, feature_dim = 768
    nega_frames = config['optim_config']['nega_frames']
    input_dim = config['model_config']['input_dim']
    feature_dim = config['model_config']['feat_dim']
    reduc_dim = config['model_config']['reduc_dim']
    d_ff = config['model_config']['d_ff']
    n_heads = config['model_config']['n_heads']
    n_language = config['model_config']['n_language']

    # model checkpoint to continue from
    model_checkpoint = None
    

