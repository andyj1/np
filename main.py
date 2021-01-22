#!/usr/bin/env python3

import argparse
import gc
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from botorch.exceptions import BadInitialCandidatesWarning
from sklearn.model_selection import train_test_split
from tqdm import trange

from acquisition import Acquisition
from dataset import getMOM4data, getTOYdata
from surrogate import SurrogateModel
from utils import bo_utils, np_utils, viz_utils
from utils.utils import (checkParamIsSentToCuda, loadReflowOven, objective,
                         reflow_oven)

from torch.utils.tensorboard import SummaryWriter

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Running on:',str(device).upper())
dtype = torch.float

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
# SEED = 42
# torch.manual_seed(SEED)

print('[INFO] garbage collect, torch emptying cache...')
gc.collect()
torch.cuda.empty_cache()

# color test
CRED = '\033[91m'
CCYAN = '\033[93m'
CBLUE = '\033[94m'
CGREEN = '\033[92m'
CEND = '\033[0m'

def parse():
    parse_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', '--TOY', action='store_true', help='sets to toy dataset')
    parser.add_argument('--load', default=None, type=str, help='path to reflow oven model')
    parser.add_argument('--np', action='store_true', help='neural process')
    args = parser.parse_args()
    
    parse_end = time.time()
    print(': took: %.3f seconds' % (parse_end - parse_start))
    return args
    
def main():
    print('='*5,'parsing',end='')
    args = parse()
    
    # load config
    cfg = yaml.load(open('config.yml', 'r'), yaml.FullLoader)
    NUM_ITER = cfg['train']['num_iter']
    NUM_TRAIN_EPOCH = cfg['train']['num_epoch']
    NUM_SAMPLES = cfg['train']['num_samples']
    MODEL = 'NP' if args.np else 'GP'
    chip = cfg['MOM4']['parttype']

    writer = SummaryWriter(f'runs/{chip}_{MODEL}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples')
    # load reflow oven regressor
    regr_multirf = loadReflowOven(args)
    input_types = {0: 'PRE -> POST', 1: 'PRE-SPI -> POST'}
    INPUT_TYPE = 0
    print('='*10, 'selecting chip:', chip)
    print('='*10, 'input type:', input_types[INPUT_TYPE])
    
    # load data
    print('='*5, 'loading data')
    if args.toy:
        print('Loading Toy data...')
        inputs, outputs = getTOYdata(cfg, model=regr_multirf)
    else:
        data_path = './data/imputed_data.csv'
        print('='*10, 'Loading MOM4 data: %s...' % data_path)
        inputs, outputs = getMOM4data(cfg)

    # objective function: to minimize the distance from  the POST to origin toward zero
    targets = torch.FloatTensor([objective(x1, x2) for x1, x2 in zip(outputs[:,0], outputs[:,1])]).unsqueeze(1)
    print('='*10, 'data sizes:', inputs.shape, targets.shape)

    post_mean_list = []
    post_var_list = []
    train_post_dist = np.linalg.norm(targets, axis=1)
    post_mean_list.append(np.mean(train_post_dist))
    post_var_list.append(np.var(train_post_dist))

    pre_mean_list = []
    pre_var_list = []
    train_pre_dist = np.linalg.norm(inputs, axis=1)
    pre_mean_list.append(np.mean(train_pre_dist))
    pre_var_list.append(np.var(train_pre_dist))

    # initialize model and likelihood
    print('='*5, 'initializaing surrogate model')
    ITER_FROM = 0
    surrogate = SurrogateModel(inputs, targets, args, cfg,
                               writer = writer, device = device, epochs=NUM_TRAIN_EPOCH)
    if args.load is not None:
        checkpoint = torch.load(args.load)
        surrogate.model.load_state_dict(checkpoint['state_dict'])
        surrogate.optimizer.load_state_dict(checkpoint['optimizer'])
        # restarts training from the last epoch (retrieved from checkpoint filename)
        ITER_FROM = int(args.load.split('.pt')[0][-1])+1
        print(f'[INFO] Loading checkpoint from epoch: [{ITER_FROM}]')
    
    surrogate.model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # check if the parameter is sent to cuda
    # checkParamIsSentToCuda([next(surrogate.model.parameters()), inputs, targets]) 

    # prepare folders for checkpoints
    torch.backends.cudnn.benchmark = True
    folders = ['ckpts','ckpts/R0402','ckpts/R0603','ckpts/R1005','ckpts/all', 'ckpts/toy']
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True, mode=0o755)

    # initial training before acquisition loop
    initial_train_start = time.time()
    print('='*5, 'training (iteration: %s)' % ITER_FROM)
    if args.np:
        print('='*5, '[INFO] initializing Neural Process')
        surrogate.fitNP(inputs, targets, cfg, toy_bool=args.toy)
    else:
        surrogate.fitGP(inputs, targets, cfg, toy_bool=args.toy, epoch=ITER_FROM)
    initial_train_end = time.time()
    print(f'[INFO] initial train time: {initial_train_end-initial_train_start:.3f} sec')

    # training loop
    fig = plt.figure()
    ax = fig.add_subplot()
    candidates_pre_dist, candidates_post_dist = [], []
    t = trange(ITER_FROM+1, NUM_ITER+1, 1)
    for iter in t:
        iter_start = time.time()

        # optimize acquisition functions and get new observations
        acq_start = time.time()
        acq_fcn = Acquisition(cfg=cfg, model=surrogate.model, device=device)
        candidate_inputs, acq_value = acq_fcn.optimize()
        acq_end = time.time()        
        
        sys.exit(0)
        
        reflowoven_start = time.time()
        candidate_outputs = reflow_oven(candidate_inputs[0][0:2].unsqueeze(0), regr_multirf)
        reflowoven_end = time.time()

        # append distance measures
        pre_dist = objective(candidate_inputs[0][0].cpu(), candidate_inputs[0][1].cpu())
        candidates_pre_dist.append(pre_dist)
        post_dist = objective(candidate_outputs[0][0], candidate_outputs[0][1])
        candidates_post_dist.append(post_dist)

        # update input and target tensors
        inputs = torch.cat([inputs, candidate_inputs], dim=0)
        new_target = torch.FloatTensor([post_dist]).unsqueeze(1).to(device)
        targets = torch.cat([targets, new_target], dim=0)

        print()
        print('='*5, 'candidate:', inputs.shape, targets.shape)

        # re-initialize the models so they are ready for fitting on next iteration and re-train
        retrain_start = time.time()
        if args.np:
            info_dict = surrogate.fitNP(inputs, targets, cfg, toy_bool=args.toy, epoch=iter)
        else :
            info_dict = surrogate.fitGP(inputs, targets, cfg, toy_bool=args.toy, epoch=iter)
        retrain_end = time.time()
        
        # update progress bar
        t.set_description(
            desc=f'[INFO] iteration {iter:>3}/{NUM_ITER-ITER_FROM} / processing time:' +
                    f'{CRED} (total): {retrain_end-iter_start:.5f} sec, {CEND}' +
                    f'(acq): {acq_end-acq_start:.5f} sec,' +
                    f'(reflow oven): {reflowoven_end-reflowoven_start:.5f} sec,' +
                    f'(retrain): {retrain_end-retrain_start:.5f} sec,' +
                    f'{CBLUE} dist: {pre_dist:.3f} -> {post_dist:.3f} {CEND}', 
            refresh=False)

        ax.scatter(candidate_inputs[0][0].cpu(), candidate_inputs[0][1].cpu(), s=10, alpha=(iter)*1/(NUM_ITER-ITER_FROM), color='r', label='_nolegend_')
        ax.scatter(candidate_outputs[0][0], candidate_outputs[0][1], s=10,alpha=(iter)*1/(NUM_ITER-ITER_FROM), color='g', label='_nolegend_')

    # Fo results' metrics
    candidates_pre_dist = np.asarray(candidates_pre_dist)
    candidates_post_dist = np.asarray(candidates_post_dist)
    post_mean_list.append(np.mean(candidates_post_dist))
    post_var_list.append(np.var(candidates_post_dist))
    pre_mean_list.append(np.mean(candidates_pre_dist))
    pre_var_list.append(np.var(candidates_pre_dist))

    ax.scatter(candidate_inputs[0][0].cpu(), candidate_inputs[0][1].cpu(), s=10, alpha=(iter)*1/(NUM_ITER-ITER_FROM), color='r', label='PRE')
    ax.scatter(candidate_outputs[0][0], candidate_outputs[0][1], s=10,alpha=(iter)*1/(NUM_ITER-ITER_FROM), color='g', label='POST')
    # print all distances at once
    for pre, post in zip(candidates_pre_dist, candidates_post_dist):
        print(f'Distance: {pre:.3f} -> {post:.3f}')

    # compare with random normal samples as pre inputs
    cfg['train']['num_samples'] = cfg['train']['num_iter']
    random_inputs, random_outputs = getTOYdata(cfg, regr_multirf)
    ax.scatter(random_outputs[:,0], random_outputs[:,1], s=10, alpha=0.3, color='b', label='Random~N(0,120)')
    random_outputs_post_dist = np.linalg.norm(random_outputs, axis=1)
    post_mean_list.append(np.mean(random_outputs_post_dist))
    post_var_list.append(np.var(random_outputs_post_dist))
    random_inputs_pre_dist = np.linalg.norm(random_inputs, axis=1)
    pre_mean_list.append(np.mean(random_inputs_pre_dist))
    pre_var_list.append(np.var(random_inputs_pre_dist))
    
    # compare with randomly sampled data as pre inputs
    random_samples, _ = getMOM4data(cfg)
    random_samples_outputs = reflow_oven(random_samples[:,0:2], regr_multirf)
    ax.scatter(random_samples_outputs[:,0], random_samples_outputs[:,1], s=10, alpha=0.3, color='magenta', label='Randomly sampled')
    random_samples_post_dist = np.linalg.norm(random_samples_outputs, axis=1)
    post_mean_list.append(np.mean(random_samples_post_dist))
    post_var_list.append(np.var(random_samples_post_dist))
    random_samples_pre_dist = np.linalg.norm(random_samples, axis=1)
    pre_mean_list.append(np.mean(random_samples_pre_dist))
    pre_var_list.append(np.var(random_samples_pre_dist))

    print(f'Pre mean  | Training_samples: {pre_mean_list[0]:4.3f}, Candidates: {pre_mean_list[1]:4.3f}, Random_Normal: {pre_mean_list[2]:4.3f}, Random_samples: {pre_mean_list[3]:4.3f}')
    print(f'Pre Var   | Training_samples: {pre_var_list[0]:4.3f}, Candidates: {pre_var_list[1]:4.3f}, Random_Normal: {pre_var_list[2]:4.3f}, Random_samples: {pre_var_list[3]:4.3f}')

    print(f'Post mean | Training_samples: {post_mean_list[0]:4.3f}, Candidates: {post_mean_list[1]:4.3f}, Random_Normal: {post_mean_list[2]:4.3f}, Random_samples: {post_mean_list[3]:4.3f}')
    print(f'Post Var  | Training_samples: {post_var_list[0]:4.3f}, Candidates: {post_var_list[1]:4.3f}, Random_Normal: {post_var_list[2]:4.3f}, Random_samples: {post_var_list[3]:4.3f}')

    ax.legend(loc='best',labels=['PRE','POST','Random Normal input POST','Randomly sampled data input POST'])
    params = {"text.color" : "blue",
          "xtick.color" : "crimson",
          "ytick.color" : "crimson"}
    plt.rcParams.update(params)
    # ax.legend(loc='lower right')
    ax.set_xlabel('x (\u03BCm)')
    ax.set_ylabel('y (\u03BCm)')
    ax.set_xlim([-120, 120])
    ax.set_ylim([-120, 120])
    ax.set_title(f'{chip} Candidate PRE -> POST')

    ''' statistics '''
    random_outputs = [objective(x1, x2) for x1, x2 in zip(random_outputs[:,0], random_outputs[:,1])]
    random_samples_outputs = [objective(x1, x2) for x1, x2 in zip(random_samples_outputs[:,0], random_samples_outputs[:,1])]
    
    candidates_pre_dist = pd.Series(candidates_pre_dist, dtype=float, name='candidate PRE').describe()
    candidates_post_dist = pd.Series(candidates_post_dist, dtype=float, name='candidate POST').describe()
    random_outputs = pd.Series(random_outputs, dtype=float, name='Random~N(0,120)').describe()
    random_samples_outputs = pd.Series(random_samples_outputs, dtype=float, name='Randomly sampled input').describe()
    stats = pd.concat([candidates_pre_dist, candidates_post_dist, random_outputs, random_samples_outputs], axis=1)
    stats.to_csv(f'./results/{chip}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples_stats.csv')

    # plt.show()
    
    if INPUT_TYPE == 0:
        fig.savefig(f'results/reflowoven_pre_all_{chip}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples_PRE_POST.png')
    elif INPUT_TYPE == 1:
        fig.savefig(f'results/reflowoven_pre_all_{chip}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples__(PRE-SPI)_POST.png')

    writer.flush()
    writer.close()
        
if __name__ == '__main__':
    main()
