#!/usr/bin/env python3

import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch

from utils.utils import reflow_oven

pd.set_option('display.max_columns', None)

# pandas dataframe to flattened tensor shape
def flatten(df): # -> torch.FloatTensor
    return torch.FloatTensor(df.to_numpy().reshape(-1,df.shape[1]))

'''
getTOYdata(): generates a set of PRE data,
              and then passes thzrough reflow oven to get POST data
'''
def getTOYdata(cfg, model):
    start_time = time.time()
    # config
    mu = cfg['toy']['mu']
    sigma = cfg['toy']['sigma']
    num_samples = cfg['train']['num_samples']    

    # pre data
    inputs = torch.normal(mean=mu, std=sigma, size=(num_samples, 2))

    # reflow oven simulation 
    outputs = reflow_oven(inputs, model)

    end_time = time.time()
    print(': took %.3f seconds' % (end_time-start_time))
    return inputs, outputs

'''
getMOM4data: returns lists of variables from random samples (count: num_samples)
'''
def getMOM4data(cfg, data_path='./data/imputed_data.csv'):
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

'''
getMOM4chipdata: retrieves dataframe for the particular chip or all chips ('chiptype')
'''
def getMOM4chipdata(parttype, data_path):
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


# switch 90 data to 0 data
def switchOrient(x90, y90):
    y0 = float(x90)
    x0 = float(-y90)
    return x0, y0 

# check generated TOY data in standalone
if __name__=='__main__':

    model_path='./reflow_oven/models/regr_multirf_pre_all.pkl'
    regr_multirf = joblib.load(model_path)

    import yaml
    with open('config.yml', 'r')  as file:
        cfg = yaml.load(file, yaml.FullLoader)
    x_pre, y_pre, x_post, y_post = getTOYdata(cfg, regr_multirf)
    
    # with np.printoptions(precision=3, suppress=True):
    #     print('x diff:', (x_post - x_post_est))
    #     print('y diff:', (y_post - y_post_est))
    
    plt.figure()
    s = 50
    a = 0.4
    plt.scatter(x_pre, y_pre, edgecolor='k',
                c="navy", s=s, marker="s", alpha=a, label="PRE")
    plt.scatter(x_post, y_post, edgecolor='k',
                c="c", s=s, marker="^", alpha=a, label='PRE')
    plt.xlabel("x (\u03BCm)")
    plt.ylabel("y")
    plt.title("Generated Toy Data")
    plt.legend()
    plt.show()

