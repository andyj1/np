import math
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from tqdm import tqdm
import time

pd.set_option('display.max_columns', None)

'''
1. define data generators
    a. toy data: random PRE from config mean and variance
    b. randomly sampled PRE from MOM4
2. output reflow oven outputs (POST)
'''

'''
getTOYdata(): generates a toy set of PRE and POST data
'''
def getTOYdata(cfg, device):
    # config
    mu = cfg['data']['mu']
    sigma = cfg['data']['sigma']
    num_samples = cfg['train']['num_samples']    

    # pre data
    x_pre = torch.normal(mean=mu, std=sigma, size=(num_samples, 1), device=device)
    y_pre = torch.normal(mean=mu, std=sigma, size=(num_samples, 1), device=device)

    # reflow oven simulation
    
    x_post, y_post = reflow_oven(x_pre, y_pre)
    assert x_pre.shape == y_pre.shape
    assert x_post.shape == y_post.shape

    return x_pre, y_pre, x_post, y_post

'''
getMOM4data: returns lists of variables from random samples (count: num_samples)
'''
def getMOM4data(cfg, device):
    # config
    MOM4dict = cfg['MOM4']
    parttype = MOM4dict['parttype']
    pre_var1 = MOM4dict['xvar_pre']
    pre_var2 = MOM4dict['yvar_pre']
    post_var1 = MOM4dict['xvar_post']
    post_var2 = MOM4dict['yvar_post']
    
    
    # load dataframe for the selected chip type
    base_path = './data'
    file = 'imputed_data.csv'
    data_path = os.path.join(base_path, file)
    chip_df = getMOM4chipdata(data_path, chiptype=parttype)
    assert chip_df is not None, 'check chip type' # if none, there is no value for that chip
    
    # random pick
    num_samples = cfg['train']['num_samples']
    
    sampled_chip_df = chip_df.sample(n=num_samples)
    x_pre = sampled_chip_df[pre_var1].to_numpy()
    y_pre = sampled_chip_df[pre_var2].to_numpy()
    x_post = sampled_chip_df[post_var1].to_numpy()
    y_post = sampled_chip_df[post_var2].to_numpy()

    x_pre = torch.FloatTensor(x_pre.reshape(-1,1), device=device)
    y_pre = torch.FloatTensor(y_pre.reshape(-1,1), device=device)
    x_post = torch.FloatTensor(x_post.reshape(-1,1), device=device)
    y_post = torch.FloatTensor(y_post.reshape(-1,1), device=device)

    return x_pre, y_pre, x_post, y_post

'''
getMOM4chipdata: retrieves dataframe for the particular chip
'''
def getMOM4chipdata(data_path='./data/MOM4_data.csv', chiptype='R1005'):
    
    # load MOM4 dataset
    print('[INFO] Loading %s...' % data_path)
    df = pd.read_csv(data_path, index_col=False).drop(['Unnamed: 0'], axis=1)
    df.reset_index(drop=True, inplace=True)
    assert df.isnull().sum().sum() == 0

    # prepare dataframes for each chip
    # chip_dataframes = edict()
    # for name, group in df.groupby(['PartType']):
    #     chip_dataframes[str(name)] = group
    
    # return dataframe for the selected chip type
    chip_df = None
    for name, group in df.groupby(['PartType']):
        if name == chiptype:
            chip_df = group
            break
    # convert 90 deg to 0
    t = tqdm(chip_df.iterrows(), total=len(chip_df))
    for idx, row in t:    
        spi_x_avg, spi_y_avg = row[['SPI_X_AVG','SPI_Y_AVG']]
        pre_x, pre_y, post_x, post_y = row[['PRE_X','PRE_Y','POST_X','POST_Y']]
        orientation = row['Orient.']
        
        # rotate 90 to 0:
        #   : slow but this is done in place of grouping again by orientation and mapping values
        if orientation == 90:
            pre_x, pre_y = switchOrient(pre_x, pre_y)
            post_x, post_y = switchOrient(pre_x, pre_y)
            spi_x_avg, spi_y_avg = switchOrient(spi_x_avg, spi_y_avg)
            # update
            chip_df.loc[idx, 'PRE_X'] = pre_x
            chip_df.loc[idx, 'PRE_Y'] = pre_y
            chip_df.loc[idx, 'POST_X'] = post_x
            chip_df.loc[idx, 'POST_Y'] = post_y
            chip_df.loc[idx, 'SPI_X_AVG'] = spi_x_avg
            chip_df.loc[idx, 'SPI_Y_AVG'] = spi_y_avg
            t.set_description(f'Switching 90 to 0 for index: {idx}')
            
    return chip_df

'''
reflow_oven: function to model reflow oven shift behavior from MultiOutput RF regressor
'''
def reflow_oven(x_pre, y_pre, model_path='./RFRegressor/models/regr_multirf.pkl'):
   
    x_pre = x_pre.cpu()
    y_pre = y_pre.cpu()
    
    # load RF regressor
    regr_multirf = joblib.load(model_path)
    # X_test: Nx2 numpy array
    x_pre = x_pre.reshape(-1,1)
    y_pre = y_pre.reshape(-1,1)
    X_test = np.concatenate((x_pre, y_pre), axis=1)

    # evaluate
    y_multirf = regr_multirf.predict(X_test)
    x_post, y_post = y_multirf[:, 0], y_multirf[:, 1]

    # print(f'[INFO] reflow oven simulation took {end_time-start_time:.3f} seconds')
    return x_post, y_post

# test standalone
if __name__=='__main__':
    import yaml
    with open('config.yml', 'r')  as file:
        cfg = yaml.load(file, yaml.FullLoader)
    x_pre, y_pre, x_post, y_post = getMOM4data(cfg)
    x_post_est, y_post_est = reflow_oven(x_pre, y_pre)
    print(x_post_est.shape, y_post_est.shape)
    print('='*67)
    print('x_post:\t\t',x_post)
    print('x_post_est:\t',x_post_est)
    print('y_post:\t\t',y_post)
    print('y_post_est:\t',y_post_est)
    print()

    with np.printoptions(precision=3, suppress=True):
        print('x diff:', (x_post - x_post_est))
        print('y diff:', (y_post - y_post_est))
    
    plt.figure()
    s = 50
    a = 0.4
    plt.scatter(x_post, y_post, edgecolor='k',
                c="navy", s=s, marker="s", alpha=a, label="Actual")
    plt.scatter(x_post_est, y_post_est, edgecolor='k',
                c="c", s=s, marker="^", alpha=a, label='Estimate')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Multi-Output RF")
    plt.legend()
    plt.show()


# switch 90 data to 0 data
def switchOrient(x90, y90):
    y0 = float(x90)
    x0 = float(-y90)
    return x0, y0 

