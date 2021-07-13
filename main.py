#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import configparser
import logging
import os
import sys

import torch
import torch.optim as optim

import data, model, trainer

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # prepare figure directory
    save_img_path = './fig'
    os.makedirs(save_img_path, exist_ok=True) if not os.path.isdir(save_img_path) else None            
            
    # configurations
    logging.basicConfig(format='[NP] %(levelname)s: %(message)s', level=logging.INFO)
    cfg = configparser.ConfigParser()
    cfg.read('config.ini')
    num_samples = int(cfg.items('dataset')[0][1])
    train_cfg = dict(zip([key for key, _ in cfg.items('train')], \
                         [int(val) if val.isdigit() \
                          else bool(val) if val=='yes' or val=='no' \
                          else [int(n.strip()) for n in val[1:-1].split(',')] if val[0]=='[' and val[-1]==']' \
                          else val for _, val in cfg.items('train')]))
    
    print('-'*20, 'configuration', '-'*20)
    print('train config:', train_cfg)
    
    # parameters from config
    context_size = train_cfg['context_size']
    n_epoch = train_cfg['n_epoch']
    test_interval = train_cfg['test_interval'] 
    input_dim = train_cfg['x_dim']
    lr = float(train_cfg['lr'])
    device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
    
    try:
        dataset_name = sys.argv[1]
    except (NameError, IndexError) as error:
        msg = f'[ERROR] Specify a dataset: [sine, parabola]'
        print(msg)
        print(f'input dim: {input_dim}')
        sys.exit(0)
    
    ''' dataset '''
    train_set, val_set = None, None
    if dataset_name == 'sine':
        assert input_dim == 1, f'input dim: {input_dim}'
        dataset = data.CustomData(input_dim=input_dim, num_samples=num_samples, type='sine') # 1-D only
    elif dataset_name == 'parabola': # sphere
        # assert isinstance(input_dim, int), f'input dim: {input_dim}'
        # assert input_dim == 2, f'input dim: {input_dim}'
        dataset = data.CustomData(input_dim=input_dim, num_samples=num_samples, type='parabola')  
    elif dataset_name == 'toy':
        dataset = data.CustomData(input_dim=input_dim, num_samples=num_samples, type='toy')
    
    
    if train_set is None and val_set is None:
        lengths = [int(0.8 * len(dataset)), int(0.2 * len(dataset))]
        train_set, val_set = torch.utils.data.random_split(dataset, lengths)
    train_set = dataset
    
    ''' dataloader '''
    # entire sequence for signal data, config batch size for mnist
    bsz = len(dataset)
    num_workers = int(torch.cuda.device_count()) * 8
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsz, shuffle=True, pin_memory=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=bsz, shuffle=True, pin_memory=False, num_workers=num_workers)
    
    ''' initialize model '''
    surrogate_model = model.AttentiveNP(train_cfg, device=device)
    # surrogate_model = model.FeedforwardMLP(train_cfg, device)
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    
    ''' train model '''
    ModelTrainer = trainer.NPTrainer(surrogate_model, context_size, optimizer, dataset_name, device)
    ModelTrainer.train(train_loader, test_loader, n_epoch, test_interval)
    