#!/usr/bin/env python3

import time
from importlib import import_module

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from utils import self_alignment as reflow_oven
from data.toy import ToyData

def flatten(df): # -> torch.FloatTensor
    # function: reshape pandas dataframe to flattened tensor shape
    return torch.FloatTensor(df.to_numpy().reshape(-1,df.shape[1]))

def getTOYdata(cfg, model=None, device='cuda'):
    '''
    getTOYdata(): generates a set of PRE data and passes through reflow oven to get POST data
    '''
    toycfg = cfg['toy']
    toy = ToyData(toycfg)
    inputs = torch.cat([toy.preLW(), toy.preAngle(), toy.SPILW(), toy.SPIcenter(), toy.SPIVolumes()], dim=1)

    # self alignment
    global method
    outputs, method = reflow_oven.self_alignment(inputs, model=None, toycfg=toycfg) # no model needed for toy
    print('[INFO] self alignment method:',method)
    
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    
    return inputs, outputs

def getSineData(cfg):
    '''
    getTOYdata(): generates a set of PRE data,
                and then passes thzrough reflow oven to get POST data
    '''
    import math

    import numpy as np

    start_time = time.time()

    num_samples = cfg['train']['num_samples']
    x_bounds = (-math.pi, math.pi)
    a_min, a_max = (-5., 5.) # amplitude range
    b_min, b_max = (-0, 0   ) # shift range    
    a = (a_max - a_min) * np.random.rand() + a_min
    b = (b_max - b_min) * np.random.rand() + b_min
    
    x = torch.linspace(x_bounds[0], x_bounds[1], num_samples).unsqueeze(1)
    y = a * torch.sin(x - b)

    end_time = time.time()
    print(': took %.3f seconds' % (end_time-start_time))
    
    return x, y

def getMOM4data(cfg, data_path='./data/imputed_data.csv'):
    '''
    getMOM4data: returns lists of variables from random samples (count: num_samples)
    '''
    data_path = "./data_analysis/('R1005', 0)_1.csv"
    start_time = time.time()
    
    MOM4dict = cfg['MOM4']
    input_var = MOM4dict['input_var']
    output_var = MOM4dict['output_var']

    # load data for the selected chip type
    parttype = cfg['MOM4']['parttype']
    chip_df = getMOM4chipdata(parttype, data_path) # datafrmae for the selected {chip} located at {data_path}
    
    # random sample `num_samples` number of data
    num_samples = cfg['train']['num_samples']
    sampled_chip_df = chip_df.sample(n=num_samples, random_state=42)
    inputs = flatten(sampled_chip_df[input_var]) # pre x only

    # for further manipulation such as PRE-SPI L,W, 
    # extract by input variables from config, subtract and concatenate columns
    # x_pre = inputs[:,0]-inputs[:,1]
    # y_pre = inputs[:,2]-inputs[:,3]
    
    outputs = flatten(sampled_chip_df[output_var])
    
    assert len(inputs) == len(outputs)
    
    end_time = time.time()
    print(': took %.3f seconds' % (end_time-start_time))
    return inputs, outputs

def getMOM4chipdata(parttype, data_path):
    '''
    getMOM4chipdata: retrieves dataframe for the particular chip or all chips ('chiptype')
    '''
    # load MOM4 dataset
    df = pd.read_csv(data_path, index_col=False).drop(['Unnamed: 0'], axis=1)
    df.reset_index(drop=True, inplace=True)
    assert df.isnull().sum().sum() == 0

    # select the dataframe for the chip type in conifg
    chip_df = None
    if parttype == 'all':
        chip_df = df
    else:
        for name, group in df.groupby(['PartType']):
            if name == parttype:
                chip_df = group
                break
    # if none, there is no value for that chip
    assert chip_df is not None, '[Error] check chip type' 
    
    return chip_df

# check generated TOY data in standalone
if __name__=='__main__':
    with open('config.yml', 'r')  as file:
        cfg = yaml.load(file, yaml.FullLoader)

    # inputs, outputs = getSineData(cfg)
    inputs, outputs = getTOYdata(cfg)
    print(inputs.shape, outputs.shape)
    inputs = inputs.cpu()
    outputs = outputs.cpu()
    
    from utils.self_alignment import *
    
    outputs = constantshift(inputs, cfg['toy'])
    print(outputs.shape)
    outputs = shift(inputs, cfg['toy'])
    print(outputs.shape)
    outputs = shiftPolar(inputs, cfg['toy'])
    print(outputs.shape)
    outputs = tensionSimple(inputs, cfg['toy'])
    print(outputs.shape)
    outputs = tension(inputs, cfg['toy'])
    print(outputs.shape)
    
    # model_path='./reflow_oven/models/regr_multirf_pre_all.pkl'
    # regr_multirf = joblib.load(model_path)
    # x_pre, y_pre, x_post, y_post = getTOYdata(cfg, regr_multirf)
    # with np.printoptions(precision=3, suppress=True):
    #     print('x diff:', (x_post - x_post_est))
    #     print('y diff:', (y_post - y_post_est))
    
    # plt.figure()
    # s = 50
    # a = 0.4
    # plt.scatter(x_pre, y_pre, edgecolor='k', c="navy", s=s, marker="s", alpha=a, label="PRE")
    # plt.scatter(x_post, y_post, edgecolor='k', c="c", s=s, marker="^", alpha=a, label='PRE')
    # plt.xlabel("x (\u03BCm)")
    # plt.ylabel("y")
    # plt.title("Generated Toy Data")
    # plt.legend()
    # plt.show()
    # plt.figure()
    # s = 50
    # a = 0.4
    # plt.scatter(inputs[:,0], inputs[:,1], edgecolor='k', c="navy", s=s, marker="s", alpha=a, label="PRE")
    # plt.scatter(outputs[:,0], outputs[:,1], edgecolor='k', c="c", s=s, marker="^", alpha=a, label='POST')
    # # plt.scatter(inputs, outputs, edgecolor='k', c="navy", s=5, marker="s", alpha=a, label="Sine")



    # plt.xlabel("x (\u03BCm)")
    # plt.ylabel("y")
    # plt.title(f'Generated Toy Data ({method})')
    # plt.legend()
    # plt.axis('equal')
    # plt.grid(linewidth=0.5)
    # plt.tight_layout()
    # plt.savefig(f'./sample_toydata_{method}.png')
    # # plt.xlim([-500, 500])
    # # plt.ylim([-500, 500])
    # # plt.show()
