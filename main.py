#!/usr/bin/env python3

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import trange

from acquisition import Acquisition
from dataset import getMOM4data, getSineData, getTOYdata
from surrogate import SurrogateModel
from utils import parse
from utils import self_alignment as reflow_oven
from utils import utils


from data.toy import ToyData
    
''' global variables '''
# color codes for text in UNIX shell
CRED = '\033[91m'
CGREEN = '\033[92m'
CCYAN = '\033[93m'
CBLUE = '\033[94m'
CEND = '\033[0m'

def setup():
    args = parse.parse_args()
    cfg = yaml.load(open('config.yml', 'r'), yaml.FullLoader)
    
    utils.supress_warnings()
    utils.set_torch_seed(seed=42, benchmark=False)
    utils.set_decomposition_type(args.cholesky) # if argument is True, then computes exact decomposition using Cholesky
    utils.clean_memory()        # garbage collection and cuda clear memory cache
    
    print('[INFO] running on:', args.device.lower())
    return args, cfg

def writer_setup(params):
    CHIP, MODEL, NUM_CANDIDATES, NUM_TRAIN_EPOCH, NUM_SAMPLES = params
    logname = f'runs/{CHIP}_{MODEL}_{NUM_CANDIDATES}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples'
    return SummaryWriter(logname)

def load_data(cfg, device):
    start_time = time.time()
    inputs, outputs = getTOYdata(cfg, model=None, device=device)  # model=regr_multirf
    targets = utils.objective(outputs[:,0:2]) # outputs: 2-dimensional
    end_time = time.time()
    print(f'[INFO] inputs: {inputs.shape}, outputs: {outputs.shape}, targets: {targets.shape}')
    print('Loading data: took %.3f seconds' % (end_time-start_time))
    
    return inputs, outputs, targets

def load_surrogate(params):
    inputs, targets, args, cfg, writer, NUM_TRAIN_EPOCH, MODEL = params
    start_time = time.time()
    surrogate = SurrogateModel(train_X=inputs, train_Y=targets, args=args, cfg=cfg, writer=writer, epochs=NUM_TRAIN_EPOCH, model_type=MODEL)
    end_time = time.time()
    print(f'[INFO] initialize model: {MODEL}, chip: {cfg["MOM4"]["parttype"]}: took %.3f seconds' % (end_time-start_time))
    return surrogate

def load_checkpoint(params):
    args, surrogate = params
    ITER_FROM = 0
    if args.load is None:
        return surrogate, ITER_FROM
    start_time = time.time()
    checkpoint = torch.load(args.load, map_location=args.device)
    surrogate.model.load_state_dict(checkpoint['state_dict'])
    surrogate.optimizer.load_state_dict(checkpoint['optimizer'])
    end_time = time.time()
    print('Loading ckpt (candidate_iter: {ITER_FROM}): took %.3f seconds' % (end_time-start_time))
    # restarts training from the last epoch (retrieved from checkpoint filename)
    ITER_FROM = int(args.load.split('.pt')[0].split('_')[-1])
    return surrogate, ITER_FROM

def uniform_dist_fcn(_min, _max):
    a, b = _max, _min  # ~U(_min, _max)
    scaler = lambda x: b + (a - b) * x
    return scaler

def main():
    print('Setting up workspace environment...')
    args, cfg = setup()
    args.device = torch.device(args.device)
    
    # define constants
    NUM_CANDIDATES = cfg['train']['num_candidates']
    NUM_TRAIN_EPOCH = cfg['train']['num_epoch']
    NUM_SAMPLES = cfg['train']['num_samples']
    MODEL = args.model.upper()  # ['GP','NP','ANP']
    CHIP = cfg['MOM4']['parttype'] if args.chip is None else args.chip
    if args.model.lower() in cfg.keys(): 
        print(f'{CRED}# context in {args.model.lower()}{CEND}:', cfg[args.model.lower()]['num_context'])
    
    # tensorboard summary writer
    writer_params = (CHIP, MODEL, NUM_CANDIDATES, NUM_TRAIN_EPOCH, NUM_SAMPLES)
    writer = writer_setup(writer_params)
    
    ''' load self alignment model (from utils) '''
    # regr_multirf, elapsed_time = loadReflowOven(args)
    # print('[INFO] loading regressor model took: %.3f seconds' % elapsed_time)

    ''' load data: inputs: [N, num_input_dim] / outputs: [N, num_input_dim] '''
    inputs, outputs, targets = load_data(cfg, args.device)
    print(inputs.shape, outputs.shape, targets.shape)
    sys.exit(1)

    ''' initialize (default) surrogate model '''    
    surrogate_params = (inputs, targets, args, cfg, writer, NUM_TRAIN_EPOCH, MODEL)
    surrogate = load_surrogate(surrogate_params)
    
    ckpt_params = (args, surrogate)
    surrogate, ITER_FROM = load_checkpoint(ckpt_params)

    # prepare folders for checkpoints
    folders = ['ckpts', 'ckpts/R0402', 'ckpts/R0603', 'ckpts/R1005', 'ckpts/all', 'ckpts/toy']
    for folder in folders: os.makedirs(folder, exist_ok=True, mode=0o755) if not os.path.isdir(folder) else None

    ''' initial training before the acquisition loop '''
    # specify which variables to use as inputs
    train_vars = [0,1] # 0,1: PRE L,W
    
    optimize_np_bool = True if args.model.upper() == 'NP' else False
    initial_train_start = time.time()
    info_dict = surrogate.train(inputs[:,train_vars], targets, args, cfg, args.toy, ITER_FROM+1)
    initial_train_end = time.time()
    print(f'initial train time: {CRED} {initial_train_end-initial_train_start:.3f} sec {CEND}')

    # ================validation=================
    # statistics for verification
    initial_input_dists = utils.objective(inputs[:, 0:2])
    initial_output_dists = utils.objective(outputs[:, 0:2])
    for num, (pre, post) in enumerate(zip(initial_input_dists, initial_output_dists)):
        print(f'# {num+1}: {pre.item():.3f} -> {post.item():.3f}')    
    # plot initial samples
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(inputs[:, 0].cpu(), inputs[:, 1].cpu(), s=5, alpha=0.1, color='orange', label='initial PRE')
    ax.scatter(outputs[:, 0].cpu(), outputs[:, 1].cpu(), s=5, alpha=0.1, color='blue', label='initial POST')
    ax.set_title(f'initial samples ({MODEL})')
    ax.grid(linewidth=0.5)
    fig.savefig(f"initial_samples.png", transparent=False)
    # ================validation=================
        
    ''' training loop '''
    toycfg = cfg['toy']
    toycfg['num_samples'] = 1
    toy = ToyData(toycfg)
        
    candidates_input_dist, candidates_output_dist = [], []
    acq_fcn = Acquisition(cfg=cfg, model=surrogate.model, device=args.device, model_type=MODEL)
    candidate_input_list, candidate_output_list = pd.DataFrame([], columns=['x','y']), pd.DataFrame([], columns=['x','y'])
    scaler = uniform_dist_fcn(0, cfg['acquisition']['bounds'])
    t = trange(1, NUM_CANDIDATES+1, 1)
    for candidate_iter in t:
        acq_start = time.time()
        # sampled shape: [1x2]
        candidate_inputs, acq_value = acq_fcn.optimize(np=optimize_np_bool) # optimize acquisition functions and get new observations
        acq_end = time.time()
        
        candidate_inputs = candidate_inputs.to(args.device)
        acq_value = acq_value.to(args.device)

        mounter_noise = scaler(torch.rand(candidate_inputs.shape))  # add mounter noise: a uniform random distribution ~U(a, b)
        candidate_inputs += mounter_noise.to(args.device)

        # self alignment simulation
        reflowoven_start = time.time()
        candidate_outputs, _ = reflow_oven.self_alignment(candidate_inputs, model=None, toycfg=cfg['toy']) # no model needed for toy
        reflowoven_end = time.time()
        # print(f'candidate #{candidate_iter}, PRE: {candidate_inputs}, POST: {candidate_outputs}')

        ''' update input and target tensors '''
        if len(candidate_inputs.shape)==3: candidate_inputs = candidate_inputs.squeeze(0)
        if len(candidate_outputs.shape)==3: candidate_outputs = candidate_outputs.squeeze(0)
        input_dist = utils.objective(candidate_inputs[:, 0:2])
        output_dist = utils.objective(candidate_outputs[:, 0:2])
        candidates_input_dist.append(input_dist)
        candidates_output_dist.append(output_dist)
        
        # generate corresponding spi info for the candidate
        preangle = toy.preAngle().to(args.device)
        spilw = toy.SPILW().to(args.device)
        spicenter = toy.SPIcenter().to(args.device)
        spivolumes = toy.SPIVolumes().to(args.device)
        candidate_input_all = torch.cat([candidate_inputs, preangle, spilw, spicenter, spivolumes], dim=1)
        
        # update candidates
        inputs = torch.cat([inputs, candidate_input_all],dim=0)
        outputs = torch.cat([outputs[:,0:2], candidate_outputs], dim=0)
        targets = torch.cat([targets, output_dist], dim=0)
        
        # store L,W inputs and outputs
        candidate_input_list.loc[candidate_iter] = candidate_inputs[0].tolist()
        candidate_output_list.loc[candidate_iter] = candidate_outputs[0].tolist()

        ''' retraining '''
        retrain_start = time.time()
        info_dict = surrogate.train(inputs[:,train_vars], targets, args, cfg, args.toy, candidate_iter)
        retrain_end = time.time()

        ''' update progress bar '''
        # msg = f"iteration {candidate_iter:>3}/{NUM_CANDIDATES-ITER_FROM} / loss: {info_dict['fopt']:.3f}," +
        #     # f'drawn candidate input: {candidate_inputs} ({candidate_inputs.shape}),' +
        #     # f'candidate output: {candidate_outputs} ({candidate_outputs.shape},' +
        #     f'\nprocessing:' +
        #     f'{CRED} total: {retrain_end-acq_start:.5f} sec, {CEND}' +
        #     # f'(acq): {acq_end-acq_start:.5f} sec,' +
        #     # f'(reflow oven): {reflowoven_end-reflowoven_start:.5f} sec,' +
        #     # f'(retrain): {retrain_end-retrain_start:.5f} sec,' +
        #     f'{CBLUE} dist: {input_dist.item():.3f} -> {output_dist.item():.3f} {CEND}\n'
        msg = f'[{candidate_iter:>3}/{NUM_CANDIDATES-ITER_FROM}] candidate #{candidate_iter}, PRE: {candidate_inputs[0][0]}, POST: {candidate_outputs[0][0]}'
        t.set_description(desc=msg, refresh=False)

        # ================validation=================        
        alpha = (candidate_iter)*1/(NUM_CANDIDATES-ITER_FROM)   # plot PRE (x,y), POST (x,y)
        ax.scatter(inputs[-1, 0].cpu(), inputs[-1, 1].cpu(), s=20, alpha=alpha, color='magenta', label='_nolegend_')
        ax.scatter(outputs[-1, 0].cpu(), outputs[-1, 1].cpu(), s=20, alpha=alpha, color='green', label='_nolegend_')
        ax.set_xlabel('x (\u03BCm)')
        ax.set_ylabel('y (\u03BCm)')
        ax.legend(loc='upper left')
        fig.savefig(f"initial_samples with candidate.png", transparent=False)
        # ================validation=================
    # ===========training loop end===================
    
    # plot final inputs and outputs (for legend purposes only)
    ax.scatter(inputs[-1, 0].cpu(), inputs[-1, 1].cpu(), s=20, alpha=alpha, color='magenta', label='sampled PRE')
    ax.scatter(outputs[-1, 0].cpu(), outputs[-1, 1].cpu(), s=20, alpha=alpha, color='green', label='sampled POST')

    # print all distances at once
    print('\nCandidate distances')
    for num, (pre, post) in enumerate(zip(candidates_input_dist, candidates_output_dist)):
        print(f'# {num+1}: {pre.item():.3f} -> {post.item():.3f}')
    print()

    # compare with randomly sampled data as pre inputs (bounded within candidate samples)
    cfg['train']['num_samples'] = len(candidates_input_dist)
    # min_input_x, max_input_x = candidate_input_list['x'].min(), candidate_input_list['x'].max()
    # min_input_y, max_input_y = candidate_input_list['y'].min(), candidate_input_list['y'].max()
    # min_input_x, max_input_x = -150, 150
    # min_input_y, max_input_y = -150, 150
    # random_x = (max_input_x - min_input_x) * torch.rand(cfg['train']['num_samples'],1, device='cpu') + min_input_x
    # random_y = (max_input_y - min_input_y) * torch.rand(cfg['train']['num_samples'],1, device='cpu') + min_input_y
    # random_samples = np.concatenate((random_x, random_y), axis=1)
    # random_samples_outputs = utils.reflow_oven(random_samples[:,0:2], regr_multirf)

    # ax.scatter(random_samples[:,0], random_samples[:,1], s=10, alpha=0.5, color='
    #           "xtick.color": "crimson", "ytick.color": "crimson"}
    # plt.rcParams.update(params)
    # ax.set_xlabel('x (\u03BCm)')
    # ax.set_ylabel('y (\u03BCm)')
    # random_samples_input_dist = np.array([objective(x,y) for x, y in zip(random_samples[:,0], random_samples[:,1])])
    # random_samples_output_dist = np.array([objective(x,y) for x, y in zip(random_samples_outputs[:,0], random_samples_outputs[:,1])])

    # ================validation=================
    # list_to_stat = [candidates_input_dist, candidates_output_dist, initial_input_dists,
    #                 initial_output_dists] #, random_samples_input_dist, random_samples_output_dist]
    # labels = ['candidates_input_dist', 'candidates_output_dist', 'initial_input_dists',
    #           'initial_output_dists']  # , 'random_samples_input_dist', 'random_samples_output_dist']

    # for idx, item in enumerate(list_to_stat):
    #     describe_item = utils.make_pd_series(item.cpu(), name=labels[idx]).describe()
    #     print(f"{labels[idx]} --> \t {CBLUE} {describe_item.loc['mean']:.4f} {CEND} +/- {describe_item.loc['std']:.4f}")
    # stats = pd.concat(list_to_stat, dtype=float, names=labels, axis=1)
    # stats.to_csv(f'./results/{CHIP}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples_stats.csv')

    # set axis (figure) attribute properties
    # ax.legend(loc='lower left',labels=['candidate PRE','candidate POST', 'random PRE','random POST'])
    # ax.legend(loc='lower left',labels=['candidate PRE','candidate POST'])
    # ================validation=================
    
    ax.set_title(f'{CHIP}, {MODEL} ({NUM_SAMPLES} initial samples)')
    ax.set_xlabel('x (\u03BCm)')
    ax.set_ylabel('y (\u03BCm)')
    ax.legend(loc='lower left')
    ax.grid(True)
    # ax.set_xlim([-120, 120])
    # ax.set_ylim([-120, 120])
    params = {"text.color": "blue", "xtick.color": "crimson", "ytick.color": "crimson"}
    plt.rcParams.update(params)

    # save figure
    image_save_path = f'results/{CHIP}_{MODEL}_{NUM_CANDIDATES}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples_{time.strftime("%Y%m%d")}.png'
    fig.savefig(image_save_path)

    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
