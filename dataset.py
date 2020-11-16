import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

pd.set_option('display.max_columns', None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

def toydataGenerator(cfg):
    # PRE (X, Y)
    x_pre, y_pre = _toydataPRE(cfg)
    x_post, y_post = _toydataPOST(x_pre, y_pre, cfg)

    assert x_pre.shape == y_pre.shape
    assert x_post.shape == y_post.shape
    print('sizes:',x_pre.shape, y_pre.shape, x_post.shape, y_post.shape)
    # print(torch.cat([x_pre, y_pre], dim=1))
    return x_pre, y_pre, x_post, y_post

def _toydataPRE(cfg):
    # PRE (X, Y)
    
    # TOY: linear sample and map to sinusoidal
    train_X = torch.linspace(0, 10, cfg['num_samples'])
    # training data needs to be explicitly multi-dimensional
    train_X = train_X.unsqueeze(1)
    # sample observed values and add some synthetic noise
    train_Y = torch.sin(train_X * (2 * math.pi)) + 0.15 * torch.randn_like(train_X)
    return train_X, train_Y

    # x_pre = torch.normal(mean=cfg['mu'], std=cfg['sigma'], size=(cfg['num_samples'], 1), dtype=dtype, device=device)
    # noise = 0.*torch.randn_like(x_pre)
    # y_pre = torch.sin(x_pre * (2*math.pi)) + noise # sinusoidal
    # return x_pre, y_pre

def _toydataPOST(x_pre, y_pre, cfg):
    # POST (X, Y) after reflow oven
    
    # TOY: random sample and map to sinusoidal
    train_X = torch.randn(cfg['num_samples'], dtype=dtype)
    # training data needs to be explicitly multi-dimensional
    train_X = train_X.unsqueeze(1)
    # sample observed values and add some synthetic noise
    train_Y = torch.sin(train_X * (2 * math.pi)) + 0.15 * torch.randn_like(train_X)
    return train_X, train_Y
    
    # TOY: sample from both normal, add shifts
    # x_post = torch.randn_like(x_pre) * cfg['dist_sigma'] + cfg['dist_mu']
    # y_post = torch.randn_like(y_pre) * cfg['dist_sigma'] + cfg['dist_mu']
    # offset_angle = torch.randn_like(x_pre) * cfg['angle_sigma'] + cfg['angle_mu']
    # x_post = x_pre + (x_post * torch.cos(offset_angle * math.pi/180))
    # y_post = y_pre + (y_post * torch.sin(offset_angle * math.pi/180))
    # return x_post, y_post

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