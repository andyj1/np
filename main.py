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
from utils import self_alignment as reflow_oven, parse, utils

''' global variables '''
# color codes for text in UNIX shell
CRED = '\033[91m'
CGREEN = '\033[92m'
CCYAN = '\033[93m'
CBLUE = '\033[94m'
CEND = '\033[0m'

def main():
    ''' preliminary setup '''
    print('Setting up workspace environment...')
    args = parse.parse_args()
    utils.supress_warnings()
    utils.set_torch_seed(seed=42, benchmark=False)
    utils.set_decomposition_type(args.cholesky) # if argument is True, then computes exact decomposition using Cholesky
    utils.clean_memory()        # garbage collection and cuda clear memory cache

    ''' load config parameters '''
    print('Loading configurations...')
    cfg = yaml.load(open('config.yml', 'r'), yaml.FullLoader)
    device = torch.device(cfg['train']['device'])
    NUM_CANDIDATES = cfg['train']['num_candidates']
    NUM_TRAIN_EPOCH = cfg['train']['num_epoch']
    NUM_SAMPLES = cfg['train']['num_samples']
    MODEL = args.model.upper()  # ['GP','NP','ANP']
    CHIP = cfg['MOM4']['parttype'] if args.chip is None else args.chip
    
    # tensorboard summary writer
    writer = SummaryWriter(
        f'runs/{CHIP}_{MODEL}_{NUM_CANDIDATES}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples')
    print('[INFO] running on:', str(device).lower())

    ''' manipulate context and target sizes: discarded '''
    if args.model.lower() in cfg.keys():
        print(f'{CRED}# Context in {args.model.lower()}{CEND}:', cfg[args.model.lower()]['num_context'])
    # cfg['acquisition']['num_restarts'] = cfg['acquisition']['raw_samples']
    # if args.model.lower() in cfg.keys():
    #     cfg[args.model.lower()]['num_context'] = 0
    # cfg[args.model.lower()].update({'num_context': (cfg['train']['num_samples'] - cfg['acquisition']['num_restarts'])})

    ''' load data: inputs: [N, num_input_dim] / outputs: [N, 2] '''
    # regr_multirf, elapsed_time = loadReflowOven(args)
    # print('[INFO] loading regressor model took: %.3f seconds' % elapsed_time)
    start_time = time.time()
    inputs, outputs = getTOYdata(cfg, model=None, device=device)  # model=regr_multirf
    targets = utils.objective(outputs) # outputs: 2-dimensional
    end_time = time.time()
    print('Loading data: took %.3f seconds' % (end_time-start_time))
    
    print(f'[INFO] inputs: {inputs.shape}, outputs: {outputs.shape}, targets: {targets.shape}')

    ''' initialize model and likelihood '''
    ITER_FROM = 0
    start_time = time.time()
    surrogate = SurrogateModel(inputs, targets, args, cfg, writer=writer, device=device, epochs=NUM_TRAIN_EPOCH, model_type=MODEL)
    end_time = time.time()
    print(f'Initializing model ({MODEL}, {CHIP}): took %.3f seconds' % (end_time-start_time))

    if args.load is not None:
        start_time = time.time()
        checkpoint = torch.load(args.load)
        surrogate.model.load_state_dict(checkpoint['state_dict'])
        surrogate.optimizer.load_state_dict(checkpoint['optimizer'])
        # restarts training from the last epoch (retrieved from checkpoint filename)
        ITER_FROM = int(args.load.split('.pt')[0].split('_')[-1])
        end_time = time.time()
        print('Loading ckpt (candidate_iter: {ITER_FROM}): took %.3f seconds' % (end_time-start_time))
    surrogate.model.to(device)

    # prepare folders for checkpoints
    folders = ['ckpts', 'ckpts/R0402', 'ckpts/R0603', 'ckpts/R1005', 'ckpts/all', 'ckpts/toy']
    for folder in folders: os.makedirs(folder, exist_ok=True, mode=0o755) if not os.path.isdir(folder) else None

    # initial training before the acquisition loop
    optimize_np_bool = False
    print('Training model...')
    initial_train_start = time.time()
    if 'GP' in MODEL:
        info_dict = surrogate.fitGP(inputs[:, 0:2], targets, cfg, toy_bool=args.toy, iter=ITER_FROM+1)
    else:
        optimize_np_bool = True
        info_dict = surrogate.fitNP(inputs[:, 0:2], targets, cfg, toy_bool=args.toy)
    initial_train_end = time.time()
    print(f'[INFO] initial train time: {CRED} {initial_train_end-initial_train_start:.3f} sec {CEND}')

    initial_input_dists = utils.objective(inputs[:, 0:2])
    initial_output_dists = utils.objective(outputs)
    for num, (pre, post) in enumerate(zip(initial_input_dists, initial_output_dists)):
        print(f'# {num+1}: {pre.item():.3f} -> {post.item():.3f}')
    
    # plot initial samples
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(inputs[:, 0].cpu(), inputs[:, 1].cpu(), s=5, alpha=0.1, color='orange', label='input PRE')
    ax.scatter(outputs[:, 0].cpu(), outputs[:, 1].cpu(), s=5, alpha=0.1, color='blue', label='input POST')
    ax.set_title(f'initial samples ({MODEL})')
    ax.grid(linewidth=0.5)
    fig.savefig(f"initial_samples.png", transparent=False)
        
    # training loop
    candidates_input_dist, candidates_output_dist = [], []

    # surrogate.model is SingleTaskGP (GP) or NeuralProcess (NP) or AttentiveNeuralProcess (ANP)
    acq_fcn = Acquisition(cfg=cfg, model=surrogate.model,
                          device=device, model_type=MODEL)
    candidate_input_list, candidate_output_list = pd.DataFrame([], columns=['x', 'y']), pd.DataFrame([], columns=['x', 'y'])
    a, b = cfg['acquisition']['bounds'], 0  # ~U(0, bound)
    scaler = lambda x: b + (a - b) * x
    t = trange(1, NUM_CANDIDATES+1, 1)
    for candidate_iter in t:
        # optimize acquisition functions and get new observations
        acq_start = time.time()
        candidate_inputs, acq_value = acq_fcn.optimize(np=optimize_np_bool)
        acq_end = time.time()
        
        candidate_inputs = candidate_inputs.to(device)
        acq_value = acq_value.to(device)

        # add mounter noise: a uniform random distribution ~U(a, b)
        mounter_noise = scaler(torch.rand(candidate_inputs.shape))
        candidate_inputs += mounter_noise.to(device)

        # reflow oven simulation
        reflowoven_start = time.time()
        candidate_outputs, _ = reflow_oven.self_alignment(candidate_inputs, model=None, toycfg=cfg['toy']) # no model needed for toy
        # candidate_outputs = reflow_oven(candidate_inputs[0][0:2].unsqueeze(0), regr_multirf)
        reflowoven_end = time.time()
        # print(f'[INFO] candidate #{candidate_iter}, PRE: {candidate_inputs}, POST: {candidate_outputs}')

        # append distance measures for candidates
        if len(candidate_inputs.shape)==3:
            candidate_inputs = candidate_inputs.squeeze(0)
        if len(candidate_outputs.shape)==3:
            candidate_outputs = candidate_outputs.squeeze(0)
        input_dist = utils.objective(candidate_inputs)
        output_dist = utils.objective(candidate_outputs)
        candidates_input_dist.append(input_dist)
        candidates_output_dist.append(output_dist)

        # update input and target tensors
        inputs = torch.cat([inputs[:, 0:2], candidate_inputs], dim=0)
        outputs = torch.cat([outputs, candidate_outputs], dim=0)
        targets = torch.cat([targets, output_dist], dim=0)

        # ax.scatter(inputs[-1, 0], inputs[-1, 1], s=20, alpha=0.5, color='magenta', label='candidate PRE')
        # ax.scatter(outputs[-1, 0], outputs[-1, 1], s=20, alpha=0.5, color='cyan', label='candidate POST')        
        # fig.savefig(f"initial_samples with candidate.png", transparent=False)
        # sys.exit(1)

        # adjust context or target size
        # ''' do either of the following '''
        # if 'NP' in MODEL:
        #     if args.not_increment_context == True:
        #         # increment context size
        #         cfg[args.model.lower()]['num_context'] += 1 # prior
        #     else:
        #         # increment target size
        #         acq_fcn.num_restarts += 1
        #         acq_fcn.raw_samples += 1
        #     print(f"[INFO] NP: context size: {cfg[args.model.lower()]['num_context']}, target size: {acq_fcn.num_restarts}")

        # re-initialize the models so they are ready for fitting on next iteration and re-train
        retrain_start = time.time()
        if 'NP' in MODEL:
            info_dict = surrogate.fitNP(inputs, targets, cfg, toy_bool=args.toy, iter=candidate_iter)
        else:
            info_dict = surrogate.fitGP(inputs, targets, cfg, toy_bool=args.toy, iter=candidate_iter)
        retrain_end = time.time()

        # update progress bar
        t.set_description(f'[{candidate_iter:>3}/{NUM_CANDIDATES-ITER_FROM}] candidate #{candidate_iter}, PRE: {candidate_inputs[0][0]}, POST: {candidate_outputs[0][0]}')

        # t.set_description(
        #     desc=f"iteration {candidate_iter:>3}/{NUM_CANDIDATES-ITER_FROM} / loss: {info_dict['fopt']:.3f}," +
        #     # f'drawn candidate input: {candidate_inputs} ({candidate_inputs.shape}),' +
        #     # f'candidate output: {candidate_outputs} ({candidate_outputs.shape},' +
        #     f'\nprocessing:' +
        #     f'{CRED} total: {retrain_end-acq_start:.5f} sec, {CEND}' +
        #     # f'(acq): {acq_end-acq_start:.5f} sec,' +
        #     # f'(reflow oven): {reflowoven_end-reflowoven_start:.5f} sec,' +
        #     # f'(retrain): {retrain_end-retrain_start:.5f} sec,' +
        #     f'{CBLUE} dist: {input_dist.item():.3f} -> {output_dist.item():.3f} {CEND}\n',
        #     refresh=False)

        # plot PRE (x,y), POST (x,y)
        alpha = (candidate_iter)*1/(NUM_CANDIDATES-ITER_FROM)
        # alpha = 1
        ax.scatter(inputs[-1, 0].cpu(), inputs[-1, 1].cpu(), s=20, alpha=alpha, color='magenta', label='_nolegend_')
        ax.scatter(outputs[-1, 0].cpu(), outputs[-1, 1].cpu(), s=20, alpha=alpha, color='green', label='_nolegend_')

        # append to candidate input/output list
        # candidate_input_list.append((candidate_inputs[0][0].cpu(),candidate_inputs[0][1].cpu()))
        # candidate_output_list.append((candidate_outputs[0][0],candidate_outputs[0][1]))

        candidate_input_list.loc[candidate_iter] = candidate_inputs[0].tolist()
        candidate_output_list.loc[candidate_iter] = candidate_outputs[0].tolist()
        
        fig.savefig(f"initial_samples with candidate.png", transparent=False)

    # plot final inputs and outputs (for legend)
    ax.scatter(inputs[-1, 0].cpu(), inputs[-1, 1].cpu(), s=20, alpha=alpha, color='magenta', label='input PRE')
    ax.scatter(outputs[-1, 0].cpu(), outputs[-1, 1].cpu(), s=20, alpha=alpha, color='green', label='input POST')

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

    # plot
    # ax.scatter(random_samples[:,0], random_samples[:,1], s=10, alpha=0.5, color='yellow', label='random PRE')
    # ax.scatter(random_samples_outputs[:,0], random_samples_outputs[:,1], s=10, alpha=1, color='blue', label='random POST')

    # random_samples_input_dist = np.array([objective(x,y) for x, y in zip(random_samples[:,0], random_samples[:,1])])
    # random_samples_output_dist = np.array([objective(x,y) for x, y in zip(random_samples_outputs[:,0], random_samples_outputs[:,1])])

    # list_to_stat = [candidates_input_dist, candidates_output_dist, initial_input_dists,
    #                 initial_output_dists] #, random_samples_input_dist, random_samples_output_dist]
    # labels = ['candidates_input_dist', 'candidates_output_dist', 'initial_input_dists',
    #           'initial_output_dists']  # , 'random_samples_input_dist', 'random_samples_output_dist']

    # ''' print summary '''
    # print('=== summary ===')
    # for idx, item in enumerate(list_to_stat):
    #     describe_item = utils.make_pd_series(item.cpu(), name=labels[idx]).describe()
    #     print(f"{labels[idx]} --> \t {CBLUE} {describe_item.loc['mean']:.4f} {CEND} +/- {describe_item.loc['std']:.4f}")
    # stats = pd.concat(list_to_stat, dtype=float, names=labels, axis=1)
    # stats.to_csv(f'./results/{CHIP}_{NUM_ITER}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples_stats.csv')

    # set axis (figure) attribute properties
    # ax.legend(loc='lower left',labels=['candidate PRE','candidate POST', 'random PRE','random POST'])
    # ax.legend(loc='lower left',labels=['candidate PRE','candidate POST'])
    
    # ax.legend(loc='lower left', labels=[
    #           'input PRE', 'input POST', 'candidate PRE', 'candidate POST'])
    # params = {"text.color": "blue",
    #           "xtick.color": "crimson", "ytick.color": "crimson"}
    # plt.rcParams.update(params)
    # ax.set_xlabel('x (\u03BCm)')
    # ax.set_ylabel('y (\u03BCm)')
    # ax.set_xlim([-120, 120])
    # ax.set_ylim([-120, 120])
    # ax.set_title(f'{CHIP}, {MODEL} ({NUM_SAMPLES} initial samples)')
    # ax.grid(True)

    # save figure
    # image_save_path = f'results/{CHIP}_{MODEL}_{NUM_CANDIDATES}iter_{NUM_TRAIN_EPOCH}epoch_{NUM_SAMPLES}samples_{time.strftime("%Y%m%d")}.png'
    # fig.savefig(image_save_path)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
