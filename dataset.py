#!/usr/bin/env python3

import time
import yaml

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch

from utils.utils import reflow_oven, test_self_alignment

pd.set_option('display.max_columns', None)

# pandas dataframe to flattened tensor shape
def flatten(df): # -> torch.FloatTensor
    return torch.FloatTensor(df.to_numpy().reshape(-1,df.shape[1]))

def getTOYdata(cfg, model=None):
    '''
    getTOYdata(): generates a set of PRE data,
                and then passes thzrough reflow oven to get POST data
    '''
    start_time = time.time()
    # config
    mu = cfg['toy']['mu']
    sigma = cfg['toy']['sigma']
    num_samples = cfg['train']['num_samples']    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # generate toy data    
    selected_chip = cfg['MOM4']['parttype']
    print('selected:',selected_chip)
    chip = cfg['MOM4']['chips'][selected_chip] # variables: length, wdith

    if model is not None:
        #   (1) pre aoi (x,y)
        inputs = torch.normal(mean=mu, std=sigma, size=(num_samples, 2))
        # reflow oven simulation from a model
        #   inputs: Pre (x,y)
        #   outputs: Post (x,y)
        outputs = reflow_oven(inputs, model)
    else:
        #   (2) pre aoi (x,y) + spi (x,y) + spi volumes 1, 2 + spi volume difference
        pre_chip = torch.normal(mean=mu, std=sigma, size=(num_samples, 2))
        pre_theta = (torch.rand(size=(num_samples,1))*15) # pre angle range
        spi_1 = pre_chip + \
            torch.rand(size=(num_samples, 2))* torch.FloatTensor([chip['length']*0.1, chip['width']*0.1]).repeat(num_samples, 1) # spi (x,y) are supposedly offset from pre (x,y) in the POSITIVE x direction here
        spi_2 = pre_chip + \
            torch.rand(size=(num_samples, 2)) * torch.FloatTensor([-chip['length']*0.1, chip['width']*0.1]).repeat(num_samples, 1) # spi (x,y) are supposedly offset from pre (x,y) in the NEGATIVE x direction here    
        volumes = torch.rand(size=(num_samples,2)) *  (1.0 - 0.7) + 0.7 # uniform(0, 10)
        volume_difference = abs(volumes[:,0] - volumes[:,1]).unsqueeze(-1)
        inputs = torch.cat([pre_chip, pre_theta, spi_1, spi_2, volumes, volume_difference], dim=1) # horizontally

        # reflow oven simulation: experiment
        #   inputs: [pre_x, pre_y, pre_theta, spi_x1, spi_y1, spi_x2, spi_y2, volume_1, volume_2, volume_difference]
        #   outputs: [post_chip, post_theta]
        simulation_option = 1
        outputs = test_self_alignment(inputs, chip=chip, chipname=selected_chip, option=simulation_option)

    end_time = time.time()
    print(': took %.3f seconds' % (end_time-start_time))
    
    
    return inputs, outputs

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


# switch 90 data to 0 data
def switchOrient(x90, y90):
    y0 = float(x90)
    x0 = float(-y90)
    return x0, y0 

# check generated TOY data in standalone
if __name__=='__main__':
    with open('config.yml', 'r')  as file:
        cfg = yaml.load(file, yaml.FullLoader)

    # model_path='./reflow_oven/models/regr_multirf_pre_all.pkl'
    # regr_multirf = joblib.load(model_path)

    # x_pre, y_pre, x_post, y_post = getTOYdata(cfg, regr_multirf)
    
    # with np.printoptions(precision=3, suppress=True):
    #     print('x diff:', (x_post - x_post_est))
    #     print('y diff:', (y_post - y_post_est))
    
    # plt.figure()
    # s = 50
    # a = 0.4
    # plt.scatter(x_pre, y_pre, edgecolor='k',
    #             c="navy", s=s, marker="s", alpha=a, label="PRE")
    # plt.scatter(x_post, y_post, edgecolor='k',
    #             c="c", s=s, marker="^", alpha=a, label='PRE')
    # plt.xlabel("x (\u03BCm)")
    # plt.ylabel("y")
    # plt.title("Generated Toy Data")
    # plt.legend()
    # plt.show()

    inputs, outputs = getTOYdata(cfg)
    print(inputs.shape, outputs.shape)
    inputs = inputs.cpu()
    outputs = outputs.cpu()
    
    plt.figure()
    s = 50
    a = 0.4
    plt.scatter(inputs[:,0], inputs[:,1], edgecolor='k',
                c="navy", s=s, marker="s", alpha=a, label="PRE")
    plt.scatter(outputs[:,0], outputs[:,1], edgecolor='k',
                c="c", s=s, marker="^", alpha=a, label='POST')
    plt.xlabel("x (\u03BCm)")
    plt.ylabel("y")
    plt.title("Generated Toy Data")
    plt.legend()
    # plt.xlim([-500, 500])
    # plt.ylim([-500, 500])
    plt.axis('equal')
    plt.grid(linewidth=0.5)
    plt.tight_layout()
    plt.savefig('./sample_toydata.png')
    # plt.show()