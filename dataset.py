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
def toydataGenerator(cfg, toy_boolean=True):
    if toy_boolean:
        x_pre, y_pre = _toydataPRE(cfg)
    else:
        x_pre, y_pre = getMOM4data(cfg=cfg, file='MOM4_data.csv')
    x_post, y_post = reflow_oven(x_pre, y_pre, cfg, toy=toy_boolean)

    assert x_pre.shape == y_pre.shape
    assert x_post.shape == y_post.shape
    print('[INFO] Generated Data:\n\tPre:', x_pre.shape, y_pre.shape, \
            '\n\tPost:', x_post.shape, y_post.shape)
    return x_pre, y_pre, x_post, y_post

''' private
_toydataPRE: generates PRE data (either random or according to a defined function)
'''
def _toydataPRE(cfg):
    x_pre = torch.normal(mean=cfg['data']['mu'], std=cfg['data']['sigma'], size=(cfg['train']['num_samples'], 1))
    y_pre = torch.normal(mean=cfg['data']['mu'], std=cfg['data']['sigma'], size=(cfg['train']['num_samples'], 1))
    return x_pre, y_pre # size: ()

'''
reflow_oven: function to model reflow oven shift behavior,
             generates POST data
'''
def reflow_oven(x_pre, y_pre, cfg, toy=True):
    if toy:
        # generate shifts in x and y direction randomly
        # - random distance shift
        x_post = torch.randn_like(x_pre) * cfg['data']['dist_sigma'] + cfg['data']['dist_mu'] # random numbers with mu and sigma
        y_post = torch.randn_like(y_pre) * cfg['data']['dist_sigma'] + cfg['data']['dist_mu']
        # - random angular shift
        offset_angle = 180 + torch.atan2(y_pre, x_pre) / math.pi * 180 + torch.randn_like(x_pre) * cfg['data']['angle_sigma']
        # offset_angle = torch.randn_like(x_pre) * cfg['data']['angle_sigma'] + cfg['data']['angle_mu']

        # apply shift
        x_post = x_pre + (x_post * torch.cos(offset_angle * math.pi/180))
        y_post = y_pre + (y_post * torch.sin(offset_angle * math.pi/180))
    else:
        # for MOM4, self alignment result
        # TODO: 
        # - try KNN clustering like 고영
        # - find other way to approximate POST points
        x_post = torch.randn_like(x_pre) * cfg['data']['dist_sigma'] + cfg['data']['dist_mu']
        y_post = torch.randn_like(y_pre) * cfg['data']['dist_sigma'] + cfg['data']['dist_mu']
        offset_angle = 180 + torch.atan2(y_pre, x_pre) / math.pi * 180 + torch.randn_like(x_pre) * cfg['data']['angle_sigma']
        x_post = x_pre + (x_post * torch.cos(offset_angle * math.pi/180))
        y_post = y_pre + (y_post * torch.sin(offset_angle * math.pi/180))
    return x_post, y_post

'''
getMOM4data: load MOM4 data
'''
def getMOM4data(cfg, file='MOM4_data.csv'):
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

    parttype = cfg['MOM4']['R1005']
    xvar_pre = cfg['MOM4']['PRE_L']
    yvar_pre = cfg['MOM4']['PRE_W']
    xvar_post = cfg['MOM4']['POST_L']
    yvar_post = cfg['MOM4']['POST_W']

    x_pre = chip_dataframes[parttype][xvar_pre].values[0:10].reshape(-1, 1).astype(dtype)
    y_pre = chip_dataframes[parttype][yvar_pre].values[0:10].reshape(-1, 1).astype(dtype)
    x_post = chip_dataframes[parttype][xvar_post].values[0:10].reshape(-1, 1).astype(dtype)
    y_post = chip_dataframes[parttype][yvar_post].values[0:10].reshape(-1, 1).astype(dtype)

    # return x_pre, y_pre, x_post, y_post
    return x_pre, y_pre # determine pre and post relationship