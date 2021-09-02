#!/usr/bin/python
# -*- coding: utf-8 -*-

# https://machinelearningmastery.com/what-is-bayesian-optimization/
import sys
import time
import warnings

import torch

import bo_utils
import parse_utils
from bayesopt import BayesianOptimization
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
    
    # configurations
    cfg, args = parse_utils.parse()
    train_cfg, data_cfg, acq_cfg = cfg['train_cfg'], cfg['data_cfg'], cfg['acq_cfg']
    
    print('-'*20, 'configuration', '-'*20)
    print('config:', cfg)
    
    device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
    print('using device:', str(device))
    
    ''' initialize and train model '''
    surrogate = SurrogateModel(train_cfg, device=device)
    
    ''' BO training loop '''
    # ---------------------------------------
    
    # Bounded region of parameter space
    pbounds = dict(zip(['x1','x2'], [tuple(acq_cfg['bounds1']), tuple(acq_cfg['bounds2'])])) # {'x1': (-10, 10), 'x2': (-10, 10)}
    
    optimizer = BayesianOptimization(
        f=bo_utils.black_box_function,  # objective to minimize
        pbounds=pbounds,                # bound within which samples are randomly selected to optimize acquisition
        verbose=2,                      # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=train_cfg['random_state'],
        surrogate=surrogate,
        cfg=cfg        
    )

    start = time.time()
    optimizer.maximize(
        init_points=train_cfg['num_samples'], # number of initial points
        n_iter=acq_cfg['num_candidates'],      # num candidates
        acq='ucb', # 'ucb', 'ei', 'poi'
        # acq='ei',
        # acq='poi',
        kappa=acq_cfg['beta'],      # beta (smaller = more exploitative)
    )
    end = time.time()
    print(f'maximizing took: {(end-start):.4f} sec')
    
    # print all probed points (non-overlapping)
    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))
    
    # probe: logs result (evaluate on this point)
    # optimizer.probe(
    #     params=optimizer.max['params'],
    #     lazy=True,
    # )
    
    candidate = (optimizer.max['params']['x1'], optimizer.max['params']['x2'])
    target = optimizer.max['target']
    print('Result (optimized):', candidate, target)
    