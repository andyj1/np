#!/usr/bin/env python3

import argparse
import os
import sys
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from acquisition import Acquisition
from dataset import getMOM4data, getTOYdata
from surrogate import SurrogateModel
from utils.utils import (checkParamIsSentToCuda, clean_memory, loadReflowOven,
                         make_pd_series, objective, reflow_oven, 
                         set_decomposition_type, set_global_params)

''' global variables '''
# color codes for text in UNIX shell
CRED    = '\033[91m'
CGREEN  = '\033[92m'
CCYAN   = '\033[93m'
CBLUE   = '\033[94m'
CEND    = '\033[0m'

''' parse arguments '''
def parse():
    parse_start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', help='sets to toy dataset')
    parser.add_argument('--load_rf', default='reflow_oven/models/regr_multirf_pre_all.pkl', type=str, help='path to reflow oven model')
    parser.add_argument('--model', default='GP', type=str, help='surrogate model type')
    parser.add_argument('--load', default=None, type=str, help='path to checkpoint [pt] file')
    parser.add_argument('--chip', default=None, type=str, help='chip part type')
    parser.add_argument('--not_increment_context', default=True, action='store_false', help='increments context size over iterations, instead of target size')
    parser.add_argument('--cholesky', default=False, action='store_true', help='sets boolean to use cholesky decomposition')
    args = parser.parse_args()
    
    parse_end = time.time(); print('parsing took: %.3f seconds' % (parse_end - parse_start))
    return args

''' main function '''
def main():
    args = parse()
    device = set_global_params()            # sets seed, device type, cudnn benchmark, suppresses warnings
    set_decomposition_type(args.cholesky)   # if argument is True, then computes exact decomposition using Cholesky
    clean_memory()                          # garbage collection and cuda clear memory cache
    
    # load config parameters
    cfg = yaml.load(open('config.yml', 'r'), yaml.FullLoader)
    NUM_ITER = cfg['train']['num_iter']
    NUM_TRAIN_EPOCH = cfg['train']['num_epoch']
    NUM_SAMPLES = cfg['train']['num_samples']
    if args.chip is None:
        CHIP = cfg['MOM4']['parttype']
    else:
        CHIP = cfg['MOM4']['parttype'] = args.chip
    MODEL = args.model.upper() # ['GP','NP','ANP']
    
    # manipulate cfg for context and target size
    cfg['acquisition']['num_restarts'] = cfg['acquisition']['raw_samples']
    if args.model.lower() in cfg.keys():
        cfg[args.model.lower()]['num_context'] = (cfg['train']['num_samples'] - cfg['acquisition']['num_restarts'])
    else:
        cfg[args.model.lower()] = {'num_context': (cfg['train']['num_samples'] - cfg['acquisition']['num_restarts'])}
    
    # tensorboard summary writer
    writer = SummaryWriter(f'runs/{CHIP}_{MODEL}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples')
    
    # load reflow oven regressor
    regr_multirf, elapsed_time = loadReflowOven(args)
    print('[INFO] loading regressor model took: %.3f seconds' % elapsed_time)
    
    # load data
    data_path = './data/imputed_data.csv'
    print(f'[INFO] loading data from {data_path}', end='')
    inputs, outputs = getTOYdata(cfg, model=regr_multirf) if args.toy else getMOM4data(cfg, data_path=data_path)
    
    # objective function: to minimize the distance from POST to origin toward zero
    targets = torch.FloatTensor([objective(x1, x2) for x1, x2 in zip(outputs[:,0], outputs[:,1])]).unsqueeze(1)

    # stats about [inputs, targets] for the surrogate model
    initial_input_dists = np.array([objective(x,y) for x, y in zip(inputs[:,0], inputs[:,1])])
    initial_output_dists = np.array([objective(x,y) for x, y in zip(outputs[:,0], outputs[:,1])])

    # initialize model and likelihood
    print(f'[INFO] loading model: {MODEL}, chip: {CHIP}')
    ITER_FROM = 0
    surrogate = SurrogateModel(inputs, targets, args, cfg,
                               writer=writer, device=device, 
                               epochs=NUM_TRAIN_EPOCH, model_type=MODEL)
    
    # load model is load path is provided
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
    assert [True, True, True] == checkParamIsSentToCuda([next(surrogate.model.parameters()), inputs, targets]) if device == 'cuda' else [False,False,False]

    # prepare folders for checkpoints
    folders = ['ckpts','ckpts/R0402','ckpts/R0603','ckpts/R1005','ckpts/all', 'ckpts/toy']
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True, mode=0o755)

    # initial training before the acquisition loop
    optimize_np_bool = False
    initial_train_start = time.time()
    if 'NP' in MODEL:
        optimize_np_bool = True
        info_dict = surrogate.fitNP(inputs, targets, cfg, toy_bool=args.toy)
    else:
        info_dict = surrogate.fitGP(inputs, targets, cfg, toy_bool=args.toy, iter=ITER_FROM)
    initial_train_end = time.time()
    print(f'[INFO] initial train time: {CRED} {initial_train_end-initial_train_start:.3f} sec {CEND}')

    # training loop
    fig = plt.figure()
    ax = fig.add_subplot()
    candidates_input_dist, candidates_output_dist = [], []
    t = trange(ITER_FROM+1, NUM_ITER+1, 1)
    
    # plot initial samples
    ax.scatter(inputs[:,0].cpu(), inputs[:,1].cpu(), s=10, alpha=0.5, color='yellow', label='input PRE')
    ax.scatter(outputs[:,0].cpu(), outputs[:,1].cpu(), s=10, alpha=0.1, color='blue', label='input POST')
    
    # surrogate.model is SingleTaskGP (GP) or NeuralProcess (NP) or AttentiveNeuralProcess (ANP)
    acq_fcn = Acquisition(cfg=cfg, model=surrogate.model, device=device, model_type=MODEL)
    candidate_input_list, candidate_output_list = pd.DataFrame([], columns=['x','y']), pd.DataFrame([], columns=['x','y'])
    for iter in t:
        # optimize acquisition functions and get new observations
        acq_start = time.time()
        candidate_inputs, acq_value = acq_fcn.optimize(np=optimize_np_bool)
        acq_end = time.time()
        
        # add mounter noise: a uniform random distribution ~U(a, b)
        # 가다가 중간에 큰 값 추가
        a, b = cfg['acquisition']['bounds'], 0 # ~U(120, 0)
        scaler = lambda x: b + (a - b) * x
        mounter_noise = scaler(torch.rand(candidate_inputs.shape))
        candidate_inputs += mounter_noise.to(device)
        
        # reflow oven simulation
        reflowoven_start = time.time()
        candidate_outputs = reflow_oven(candidate_inputs[0][0:2].unsqueeze(0), regr_multirf)
        reflowoven_end = time.time()
        
        print(f'\n\n[INFO] candidate #{iter}, PRE: {candidate_inputs}, POST: {candidate_outputs}')

        # append distance measures
        input_dist = objective(candidate_inputs[0][0].cpu(), candidate_inputs[0][1].cpu())
        output_dist = objective(candidate_outputs[0][0], candidate_outputs[0][1])
        
        # store distances for statistics
        candidates_input_dist.append(input_dist)
        candidates_output_dist.append(output_dist)

        # update input and target tensors
        inputs = torch.cat([inputs, candidate_inputs], dim=0)
        new_target = torch.FloatTensor([output_dist]).unsqueeze(1).to(device)
        targets = torch.cat([targets, new_target], dim=0)
        
        # adjust context or target size
        ''' do either of the following '''
        if 'NP' in MODEL:
            if args.not_increment_context == True:
                # increment context size
                cfg[args.model.lower()]['num_context'] += 1 # prior
            else:
                # increment target size
                acq_fcn.num_restarts += 1
                acq_fcn.raw_samples += 1
            print(f"[INFO] NP: context size: {cfg[args.model.lower()]['num_context']}, target size: {acq_fcn.num_restarts}")
        
        # re-initialize the models so they are ready for fitting on next iteration and re-train
        retrain_start = time.time()
        if 'NP' in MODEL:
            info_dict = surrogate.fitNP(inputs, targets, cfg, toy_bool=args.toy, iter=iter)
        else:
            info_dict = surrogate.fitGP(inputs, targets, cfg, toy_bool=args.toy, iter=iter)
        retrain_end = time.time()
        
        # update progress bar
        t.set_description(
            desc=f"iteration {iter:>3}/{NUM_ITER-ITER_FROM} / loss: {info_dict['fopt']:.3f}," +
                    # f'drawn candidate input: {candidate_inputs} ({candidate_inputs.shape}),' +
                    # f'candidate output: {candidate_outputs} ({candidate_outputs.shape},' +
                    f'\nprocessing:' +
                    f'{CRED} total: {retrain_end-acq_start:.5f} sec, {CEND}' +
                    # f'(acq): {acq_end-acq_start:.5f} sec,' +
                    # f'(reflow oven): {reflowoven_end-reflowoven_start:.5f} sec,' +
                    # f'(retrain): {retrain_end-retrain_start:.5f} sec,' +
                    f'{CBLUE} dist: {input_dist:.3f} -> {output_dist:.3f} {CEND}', 
            refresh=False)
        
        # plot PRE (x,y), POST (x,y)
        alpha = (iter)*1/(NUM_ITER-ITER_FROM) if iter > 1 else 1
        # alpha = 1
        ax.scatter(candidate_inputs[0][0].cpu(), candidate_inputs[0][1].cpu(), s=10, alpha=alpha, color='red', label='_nolegend_')
        ax.scatter(candidate_outputs[0][0], candidate_outputs[0][1], s=10, alpha=alpha, color='green', label='_nolegend_')

        # append to candidate input/output list
        # candidate_input_list.append((candidate_inputs[0][0].cpu(),candidate_inputs[0][1].cpu()))
        # candidate_output_list.append((candidate_outputs[0][0],candidate_outputs[0][1]))
        
        candidate_input_list.loc[iter-1] = [candidate_inputs[0][0].cpu().item(), candidate_inputs[0][1].cpu().item()]
        candidate_output_list.loc[iter-1] = [candidate_outputs[0][0].item(), candidate_outputs[0][1].item()]
    
    # plot final inputs and outputs (for legend)
    ax.scatter(candidate_inputs[0][0].cpu(), candidate_inputs[0][1].cpu(), alpha=1, s=10, color='red', label='candidate PRE')
    ax.scatter(candidate_outputs[0][0], candidate_outputs[0][1], alpha=1, s=10, color='green', label='candidate POST')
    
    # print all distances at once
    print('\nCandidate distances')
    for num, (pre, post) in enumerate(zip(candidates_input_dist, candidates_output_dist)):
        print(f'# {num}: {pre:.3f} -> {post:.3f}')
    print()
    
    # compare with randomly sampled data as pre inputs (bounded within candidate samples)
    cfg['train']['num_samples'] = len(candidates_input_dist)
    min_input_x, max_input_x = candidate_input_list['x'].min(), candidate_input_list['x'].max()
    min_input_y, max_input_y = candidate_input_list['y'].min(), candidate_input_list['y'].max()
    random_x = (max_input_x - min_input_x) * torch.rand(cfg['train']['num_samples'],1, device='cpu') + min_input_x
    random_y = (max_input_y - min_input_y) * torch.rand(cfg['train']['num_samples'],1, device='cpu') + min_input_y
    random_samples = np.concatenate((random_x, random_y), axis=1)
    random_samples_outputs = reflow_oven(random_samples[:,0:2], regr_multirf)
    
    # plot 
    # ax.scatter(random_samples[:,0], random_samples[:,1], s=10, alpha=0.5, color='yellow', label='random PRE')
    # ax.scatter(random_samples_outputs[:,0], random_samples_outputs[:,1], s=10, alpha=1, color='blue', label='random POST')
    
    # print stats
    candidates_input_dist = np.asarray(candidates_input_dist)
    candidates_output_dist = np.asarray(candidates_output_dist)
    # # initial_input_dists
    # # initial_output_dists
    random_samples_input_dist = np.array([objective(x,y) for x, y in zip(random_samples[:,0], random_samples[:,1])])
    random_samples_output_dist = np.array([objective(x,y) for x, y in zip(random_samples_outputs[:,0], random_samples_outputs[:,1])])
    
    list_to_stat = [candidates_input_dist, candidates_output_dist, initial_input_dists, initial_output_dists, random_samples_input_dist, random_samples_output_dist]
    labels = ['candidates_input_dist', 'candidates_output_dist', 'initial_input_dists', 'initial_output_dists', 'random_samples_input_dist', 'random_samples_output_dist']
    print('=== summary ===')
    for idx, item in enumerate(list_to_stat):
        describe_item = make_pd_series(item, name=labels[idx]).describe()
        print(f"{labels[idx]} --> \t {CBLUE} {describe_item.loc['mean']:.4f} {CEND} +/- {describe_item.loc['std']:.4f}")
    # stats = pd.concat(list_to_stat, dtype=float, names=labels, axis=1)
    # stats.to_csv(f'./results/{CHIP}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples_stats.csv')
        
    # set axis (figure) attribute properties
    # ax.legend(loc='lower left',labels=['candidate PRE','candidate POST', 'random PRE','random POST'])
    # ax.legend(loc='lower left',labels=['candidate PRE','candidate POST'])
    ax.legend(loc='lower left',labels=['candidate PRE','candidate POST', 'input PRE','input POST'])
    params = {"text.color" : "blue",
          "xtick.color" : "crimson",
          "ytick.color" : "crimson"}
    plt.rcParams.update(params)
    ax.set_xlabel('x (\u03BCm)')
    ax.set_ylabel('y (\u03BCm)')
    ax.set_xlim([-120, 120])
    ax.set_ylim([-120, 120])
    ax.set_title(f'{CHIP}, {MODEL} ({NUM_SAMPLES} initial samples)')
    ax.grid(True)
    
    # save figure
    context_or_target = 'context++' if args.not_increment_context else 'target++'
    image_save_path = f'results/{CHIP}_{MODEL}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples'
    if 'NP' in MODEL:
         image_save_path += f'_{context_or_target}.png'
    else:
        image_save_path += '.png'
    fig.savefig(image_save_path)
    
    writer.flush()
    writer.close()
        
if __name__ == '__main__':
    main()
