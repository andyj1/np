#!/usr/bin/python
# -*- coding: utf-8 -*-

# https://machinelearningmastery.com/what-is-bayesian-optimization/
import sys
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import bo_utils
import data
import model
import parse_utils
import trainer
from acquisition import Acquisition
from surrogate import SurrogateModel

warnings.filterwarnings("ignore")

''' global variables '''
# color codes for text in UNIX shell
CRED = '\033[91m'
CGREEN = '\033[92m'
CCYAN = '\033[93m'
CBLUE = '\033[94m'
CEND = '\033[0m'

if __name__ == "__main__":
    bo_utils.setup()
    
    # prepare figure directory
            
    # configurations
    train_cfg, data_cfg, acq_cfg, args = parse_utils.parse()
    
    print('-'*20, 'configuration', '-'*20)
    print('train config:', train_cfg)
    
    # parameters from config
    num_samples = train_cfg['num_samples']
    context_size = train_cfg['context_size']
    n_epoch = train_cfg['n_epoch']
    test_interval = train_cfg['test_interval'] 
    input_dim = train_cfg['x_dim']
    lr = float(train_cfg['lr'])
    
    device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
        
    ''' dataset '''
    dataset = data.CustomData(input_dim=input_dim, 
                              num_samples=num_samples, 
                              type=args.dataset.lower(), 
                              cfg=data_cfg)  
    
    train_set, val_set = None, None
    # if train_set is None and val_set is None:
        # lengths = [int(0.8 * len(dataset)), int(0.2 * len(dataset))]
        # train_set, val_set = torch.utils.data.random_split(dataset, lengths)
        
    # set train data only for now
    train_set = dataset
    
    ''' prepare dataloader '''
    # entire sequence for signal data, config batch size for mnist
    bsz = len(dataset)
    train_loader, test_loader = None, None
    num_workers = int(torch.cuda.device_count()) * 8
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsz, shuffle=True, pin_memory=False, num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(val_set, batch_size=bsz, shuffle=True, pin_memory=False, num_workers=num_workers)
    
    ''' initialize and train model '''
    fig = plt.figure() # to plot during training steps
    surrogate = SurrogateModel(cfg=train_cfg, model_type=train_cfg['model_type'], device=device)
    surrogate.train(fig, train_loader, test_loader)
    
    ''' mounter noise: uniformly distributed ~U(a, b) '''
    scaler = lambda x, a, b: b + (a - b) * x
    mounter_noise_min, mounter_noise_max = acq_cfg['bounds']
    
    ''' BO training loop '''
    acq_fcn = Acquisition(cfg=acq_cfg, model=surrogate.model, device=device)
    candidates_inputs, candidates_outputs = [], []
    num_candidates = int(acq_cfg['num_candidates'])
    t = tqdm(range(1, num_candidates+1, 1), total=num_candidates)
    for candidate_iter in t:
        candidate_inputs, acq_value = acq_fcn.optimize()
        
        print('candidate:', candidate_inputs.cpu().detach().numpy()[0], ' objective value:', acq_value.cpu().detach().numpy())
        
        mounter_noise = scaler(torch.rand(candidate_inputs.shape), 
                               mounter_noise_min, 
                               mounter_noise_max)
        candidate_inputs += mounter_noise.to(device)

        # append candidate and acq_value to dataset
        # np.c_: fastest concatenation among [c_, stack, vstack, column_stack, concatenate]
        train_loader.dataset.x = np.c_['0', train_loader.dataset.x, candidate_inputs.cpu().detach().numpy()]
        train_loader.dataset.y = np.c_['0', train_loader.dataset.y, acq_value.cpu().detach().numpy()]
        # print(train_loader.dataset.x.shape, candidate_inputs.cpu().detach().numpy().shape)
        
    print('re-training...')
    surrogate.train(fig, train_loader, None)
    for candidate_iter in range(5):
        candidate_inputs, acq_value = acq_fcn.optimize()
        print('candidate:', candidate_inputs.cpu().detach().numpy()[0], ' objective value:', acq_value.cpu().detach().numpy())
        
        
    #     # self alignment simulation
    #     reflowoven_start = time.time()
    #     candidate_outputs, _ = reflow_oven.self_alignment(candidate_inputs, model=None, toycfg=cfg['toy']) # no model needed for toy
    #     reflowoven_end = time.time()
    #     # print(f'candidate #{candidate_iter}, PRE: {candidate_inputs}, POST: {candidate_outputs}')
