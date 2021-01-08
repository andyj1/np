#!/usr/bin/env python3

import argparse
import gc
import math
import os
import sys
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from botorch.exceptions import BadInitialCandidatesWarning
from sklearn.model_selection import train_test_split
from tqdm import trange

from acquisition import Acquisition
from dataset import getMOM4data, getTOYdata, reflow_oven
from surrogate import SurrogateModel
from utils import (bo_utils, np_utils,  # viz_utils: contourplot, draw_graphs
                   viz_utils)

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float
obj = bo_utils.objective

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
SEED = 42
torch.manual_seed(SEED)

# print('[INFO] garbage collect, torch emptying cache...')
# gc.collect()
# torch.cuda.empty_cache()
print('Running on:',device)

''' configuration '''
TEST_SIZE = 10
INPUT_TYPE = 0 # 0: PRE, 1: PRE-SPI
REFLOW_OVEN_ALL = True

def loadReflowOven(chip='R0402', inputtype=0, reflowoven=False):
    # chip: chip type for selecting the model
    # inputtype: PRE or PRE-SPI for (x,y) positions
    
    ''' load RF regressor '''
    print('='*10, 'loading regressor model', '='*10, end='')
    loadRFRegressor_start = time.time()
    model = -1
    # select a reflow oven model
    if chip == 'R0402' and inputtype == 0:
        model = 0
    elif chip == 'R0402' and inputtype == 1:
        model = 1
    elif chip == 'R0603' and inputtype == 0:
        model = 2
    elif chip == 'R0603' and inputtype == 1:
        model = 3
    elif chip == 'R1005' and inputtype == 0:
        model = 4
    elif chip == 'R1005' and inputtype == 1:
        model = 5
    elif chip == 'all' and inputtype == 0:
        model = 6
    elif chip == 'all' and inputtype == 1:
        model = 7
    
    # override for the case: each chip data with all chip data trained reflow oven model
    if reflowoven == True:
        model = 6 # PRE
        # model = 7 # PRE-SPI
    
    base_path = 'RFRegressor/models/'
    # reflow oven maps in 2 different ways: (1) PRE-SPI -> POST, or (2) PRE -> POST 
    model_paths = ['regr_multirf_R0402.pkl', 'regr_multirf_R0402_PRE-SPI.pkl', \
                    'regr_multirf_R0603.pkl', 'regr_multirf_R0603_PRE-SPI.pkl', \
                    'regr_multirf_R1005.pkl', 'regr_multirf_R1005_PRE-SPI.pkl', \
                    'regr_multirf.pkl', 'regr_multirf_PRE-SPI.pkl']
    # regr_multirf_list = []
    # for model_path in model_paths:
    #     model_path = os.path.join(base_path, model_path)
    #     regr_multirf_list.append(joblib.load(model_path))
    regr_multirf = joblib.load(os.path.join(base_path,model_paths[model]))

    loadRFRegressor_end = time.time()
    print(': took: %.3f seconds' % (loadRFRegressor_end - loadRFRegressor_start))
    return regr_multirf

def parse():
    print('='*10,'parsing','='*10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', '--TOY', action='store_true', help='sets to toy dataset')
    parser.add_argument('--load', help='destination to model dict')
    parser.add_argument('--np', action='store_true', help='neural process')
    args = parser.parse_args()
    
    # usage: main.py [--toy] [--np] [--load path/to/checkpoint]
    return args

def checkParamIsSentToCuda(args):
    for i, arg in enumerate(args):
        print(f'{i}: Cuda: {arg.is_cuda}')
            
def main():
    args = parse()
    
    # load config
    cfg = yaml.load(open('config.yml', 'r'), yaml.FullLoader)
    NUM_ITER = cfg['train']['num_iter']
    NUM_TRAIN_EPOCH = cfg['train']['num_iter']
    NUM_SAMPLES = cfg['train']['num_samples']
    chip = cfg['MOM4']['parttype']
    
    # load reflow oven regressor
    regr_multirf = loadReflowOven(chip, inputtype=INPUT_TYPE, reflowoven=REFLOW_OVEN_ALL)
    input_types = {0: 'PRE -> POST', 1: 'PRE-SPI -> POST'}
    print('='*10, 'selecting chip:', chip, ' / input type:', input_types[INPUT_TYPE])
    
    # load data
    print('='*10, 'loading data', '='*10)
    if args.toy:
        print('Loading Toy data...')
        x_pre, y_pre, x_post, y_post = getTOYdata(cfg, device, model=regr_multirf)
    else:
        print('Loading MOM4 data...')
        x_pre, y_pre, x_post, y_post = getMOM4data(cfg, device)
    
    # split into train and test
    x_pre_train, y_pre_train, x_post_train, y_post_train = \
        torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    if TEST_SIZE == 0:
        x_pre_train, y_pre_train, x_post_train, y_post_train = x_pre, y_pre, x_post, y_post
        x_pre_test, y_pre_test, x_post_test, y_post_test = torch.Tensor([]), torch.Tensor([])
    elif TEST_SIZE > 0:
        x_pre_train, x_pre_test, y_pre_train, y_pre_test = \
            train_test_split(x_pre, y_pre, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
        x_post_train, x_post_test, y_post_train, y_post_test = \
            train_test_split(x_post, y_post, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
    
    # objective function: to minimize the distance from the POST to origin toward zero
    euclidean_dist = [obj(x1, x2) for x1, x2 in zip(x_post_train, y_post_train)]

    # make into proper dimension for SurrogateModel (e.g., SingleTaskGP)
    input_tensor = torch.cat([x_pre_train, y_pre_train], dim=1)     # (N,2) dim
    target_tensor = torch.FloatTensor(euclidean_dist).unsqueeze(1)  # (N,1) dim

    # initialize model and likelihood
    ITER_FROM = 0
    surrogate = SurrogateModel(input_tensor, target_tensor, device, epochs=NUM_TRAIN_EPOCH)
    if args.load is not None:
        checkpoint = torch.load(args.load)
        surrogate.model.load_state_dict(checkpoint['state_dict'])
        surrogate.optimizer.load_state_dict(checkpoint['optimizer'])
        # restarts training from the last epoch (retrieved from checkpoint filename)
        ITER_FROM = int(args.load.split('.pt')[0][-1]) 
        print(f'[INFO] Loading checkpoint from epoch: [{ITER_FROM}]')
    
    surrogate.model.to(device)
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    
    # check if the parameter is sent to cuda
    # checkParamIsSentToCuda([next(surrogate.model.parameters()), input_tensor, target_tensor]) 
    
    # initial training before acquisition loop
    initial_train_start = time.time()
    if args.np:
        surrogate.fitNP(input_tensor, target_tensor, cfg, toy_bool=args.toy, epoch=-1)
    else:
        surrogate.fitGP(input_tensor, target_tensor, cfg, toy_bool=args.toy, epoch=-1)
    initial_train_end = time.time()
    print(f'[INFO] initial train time: {initial_train_end-initial_train_start:.3f} sec')
        
    torch.backends.cudnn.benchmark = True
    # make folders for checkpoints
    folders = ['ckpts','ckpts/R0402','ckpts/R0603','ckpts/R1005','ckpts/all', 'ckpts/toy']
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True, mode=0o755)

    # training loop
    fig = plt.figure()
    ax = fig.add_subplot()
    candidates_pre, candidates_post = [], []
    t = trange(ITER_FROM, NUM_ITER, 1)
    for iter in t:
        iter_start = time.time()

        # optimize acquisition functions and get new observations
        acq_start = time.time()
        acq_fcn = Acquisition(cfg=cfg, model=surrogate.model, device=device)
        candidate, acq_value = acq_fcn.optimize()
        acq_end = time.time()        

        # actual values from the objective, compute the distance
        x_new_pre, y_new_pre = candidate[0]  # unravel tensor to numpy floats
        x_new_pre = x_new_pre.cpu()
        y_new_pre = y_new_pre.cpu()
        
        reflowoven_start = time.time()
        x_new_post, y_new_post = reflow_oven(x_new_pre, y_new_pre, regr_multirf)
        reflowoven_end = time.time()

        pre_dist = obj(x_new_pre, y_new_pre)
        post_dist = obj(x_new_post, y_new_post)
        # append distance measures
        candidates_pre.append(pre_dist)
        candidates_post.append(post_dist)

        # update input and target tensors
        input_tensor = torch.cat([input_tensor, candidate], dim=0)
        new_target = torch.from_numpy(np.array([post_dist])).unsqueeze(1).to(device)
        target_tensor = torch.cat([target_tensor, new_target], dim=0)

        # input_tensor = input_tensor.to(device)
        # target_tensor = target_tensor.to(device)

        # re-initialize the models so they are ready for fitting on next iteration
        # and re-train
        retrain_start = time.time()
        surrogate.fitGP(input_tensor, target_tensor, cfg, toy_bool=args.toy, epoch=iter)
        retrain_end = time.time()
        
        # epoch_end = time.time()
        # update progress bar
        CRED = '\033[91m'
        CEND = '\033[0m'
        t.set_description(
            desc=f'[INFO] Epoch {iter+1} / processing time: {CRED} (total): {retrain_end-iter_start:.5f} sec, (acq):{acq_end-acq_start:.5f} sec, (reflow oven): {reflowoven_end-reflowoven_start:.5f} sec, (retrain): {retrain_end-retrain_start:.5f} sec, {CEND} pre_dist: {pre_dist:.3f}, post_dist: {post_dist:.3f}\n', 
            refresh=False)

        ax.scatter(x_new_pre, y_new_pre, s=10, alpha=(iter+1)*1/NUM_ITER, color='r')
        # ax.scatter(x_new_post, y_new_post, s=20,alpha=(iter+1)*1/NUM_ITER, color='b', label='POST')
    # ax.legend(loc='best',labels=['PRE','POST'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([-120, 120])
    ax.set_ylim([-120, 120])
    ax.set_title('Pre -> Post (darker color: later iteration)')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.scatter(x_pre, y_pre, color='g')
    ax2.legend(loc='best')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim([-120, 120])
    ax2.set_ylim([-120, 120])
    ax2.set_title('Actual Pre x,y vs. candidate x, y')
    plt.show()
    
    sys.exit()
    if INPUT_TYPE == 0:
        fig.savefig(f'reflowovenall_{chip}_{NUM_SAMPLES}_samples_PRE_POST.png')
    elif INPUT_TYPE == 1:
        fig.savefig(f'reflowovenall_{chip}_{NUM_SAMPLES}_samples_(PRE-SPI)_POST.png')

    for pre, post in zip(candidates_pre, candidates_post):
        print(f'Distance: {pre:.3f} -> {post:.3f}')
        
if __name__ == '__main__':
    main()
