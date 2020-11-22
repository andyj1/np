import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

pd.set_option('display.max_columns', None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

'''
toydataGenerator: generates a toy set of PRE and POST data
'''
def toydataGenerator(cfg):
    # PRE (X, Y)
    x_pre, y_pre = _toydataPRE(cfg)                     # random PRE
    x_post, y_post = reflow_oven(x_pre, y_pre, cfg, toy=True)    # shifted

    assert x_pre.shape == y_pre.shape
    assert x_post.shape == y_post.shape
    print('[INFO] data sizes:', \
            x_pre.shape, y_pre.shape, \
            x_post.shape, y_post.shape)
    return x_pre, y_pre, x_post, y_post

'''
_toydataPRE: generates PRE data (either random or according to some other defined function)
'''
def _toydataPRE(cfg):
    # PRE (X, Y)
    
    # 1. linear sample and map to sinusoidal, make multi-dim, add noise
    # train_X = torch.linspace(0, 10, cfg['num_samples']).unsqueeze(1)
    # train_Y = torch.sin(train_X * (2 * math.pi)) + 0.15 * torch.randn_like(train_X)
    # return train_X, train_Y

    # 2. random
    x_pre = torch.normal(mean=cfg['mu'], std=cfg['sigma'], size=(cfg['num_samples'], 1))
    y_pre = torch.normal(mean=cfg['mu'], std=cfg['sigma'], size=(cfg['num_samples'], 1))
    return x_pre, y_pre

'''
reflow_oven: function to model reflow oven shift behavior,
             generates POST data
'''
def reflow_oven(x_pre, y_pre, cfg, toy=True):
    if toy:
        # for toy data, manually add shifts
        x_post = torch.randn_like(x_pre) * cfg['dist_sigma'] + cfg['dist_mu']
        y_post = torch.randn_like(y_pre) * cfg['dist_sigma'] + cfg['dist_mu']
        offset_angle = torch.randn_like(x_pre) * cfg['angle_sigma'] + cfg['angle_mu']
        x_post = x_pre + (x_post * torch.cos(offset_angle * math.pi/180))
        y_post = y_pre + (y_post * torch.sin(offset_angle * math.pi/180))
    else:
        # for real data, self alignment result
        # TODO: 
        # - try KNN clustering like 고영
        # - find other way to approximate POST points
        x_post = torch.randn_like(x_pre) * cfg['dist_sigma'] + cfg['dist_mu']
        y_post = torch.randn_like(y_pre) * cfg['dist_sigma'] + cfg['dist_mu']
        offset_angle = 180 + torch.atan2(y_pre, x_pre) / math.pi * 180 + torch.randn_like(x_pre) * cfg['angle_sigma']
        x_post = x_pre + (x_post * torch.cos(offset_angle * math.pi/180))
        y_post = y_pre + (y_post * torch.sin(offset_angle * math.pi/180))
    return x_post, y_post

'''
getMOM4data: load MOM4 data
'''
def getMOM4data(file='MOM4_data.csv'):
    base_path = './'
    data_path = os.path.join(base_path, file)

    # prepare result directory
    result_path = os.path.join(base_path, 'result')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    # read in data
    print('[INFO] Loading %s...' % data_path)
    df = pd.read_csv(data_path, index_col=False).drop(['Unnamed: 0'], axis=1)
    df.reset_index(drop=True, inplace=True)
    assert df.isnull().sum().sum() == 0
    chip_dataframes = dict()
    for name, group in df.groupby(['PartType']):
        chip_dataframes[str(name)] = group
    parttype = 'R1005'
    xvar_pre = 'PRE_L'
    yvar_pre = 'PRE_W'
    xvar_post = 'POST_L'
    yvar_post = 'POST_W'
    data = chip_dataframes[parttype]

    x_pre = data[xvar_pre].values[0:10].reshape(-1, 1).astype(dtype)
    y_pre = data[yvar_pre].values[0:10].reshape(-1, 1).astype(dtype)
    x_post = data[xvar_post].values[0:10].reshape(-1, 1).astype(dtype)
    y_post = data[yvar_post].values[0:10].reshape(-1, 1).astype(dtype)

    return x_pre, y_pre, x_post, y_post