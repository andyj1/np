#!/usr/bin/env python3
import configparser
import matplotlib.pyplot as plt
import torch

from toy import self_alignment

cfg = configparser.ConfigParser()
cfg.read('config.ini')

stripped = cfg.items('train')[-1][1].strip('[],')
mounter_noise_min, mounter_noise_max = int(stripped[0]), int(stripped[3:])

# mounter noise: uniformly distributed ~U(a, b)
scaler = lambda x, a, b: b + (a - b) * x

def f(x, y):
    # x, y: vector (each of which is sample size)
    x_grid_pre, y_grid_pre = torch.meshgrid(x, y)
    
    # self-alignment
    if len(x.shape) != 2:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
    xy = torch.cat((x, y), 1)
    xy_shifted, method = self_alignment.self_alignment(xy)
    x_shifted, y_shifted = torch.chunk(xy_shifted, chunks=2, dim=1)
    x_shifted = x_shifted.squeeze()
    y_shifted = y_shifted.squeeze()
    
    # add mounter noise
    # x_shifted += scaler(torch.rand(x_shifted.shape), mounter_noise_min, mounter_noise_max)
    # y_shifted += scaler(torch.rand(y_shifted.shape), mounter_noise_min, mounter_noise_max)
    
    # prepare grid space
    x_grid_post, y_grid_post = torch.meshgrid(x_shifted, y_shifted)
    distances = torch.zeros(x_grid_pre.shape, dtype=torch.float32)
    
    for i in range(x_grid_pre.shape[0]):
        for j in range(x_grid_pre.shape[1]):
            # append POST distance
            point = torch.FloatTensor([x_grid_post[i,j], y_grid_post[i,j]])
            distances[i,j] = torch.linalg.norm(point, dtype=torch.float64)
    
    # distance: objective over meshgrid space
    # return distances as-is, because we are showing it as a gradient map
    return distances

def plot_grid(ax, x, y, pbounds, num_dim, model_type, iteration=None):
    x_min, x_max = pbounds[0]
    y_min, y_max = pbounds[1]
    
    if torch.is_tensor(x) and torch.is_tensor(y):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
    
    steps = 100
    x_linspace = torch.linspace(x_min, x_max, steps)
    y_linspace = torch.linspace(y_min, y_max, steps)
    z = f(x_linspace, y_linspace)

    # fig, ax = plt.subplots()
    z_min, z_max = z.min(), z.max() #-np.abs(z).max(), np.abs(z).max()
    # cmaps: 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    c = plt.imshow(z, cmap ='Spectral', 
                vmin = z_min,
                vmax = z_max, extent =[x_linspace.min(),
                                        x_linspace.max(),
                                        y_linspace.min(),
                                        y_linspace.max()],
                interpolation ='nearest', 
                origin ='lower',
                label='density')

    # ax.scatter(-2,-2, s=50, marker='x', c='r', label='candidate')
    # fig.colorbar(c)
    plt.colorbar()
    
    # ax.set_title('example')
    # ax.legend(loc='best')
    if iteration is not None:
        ax.set_title(f'{num_dim}-D {model_type.upper()}: minimize POST distance ({iteration})')
    else:
        ax.set_title(f'{num_dim}-D {model_type.upper()}: minimize POST distance')
    return ax


if __name__ == '__main__':
    from datasets.reflow_soldering.create_self_alignment_model import customMOM4chipsample
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv('./datasets/spi_clustered.csv').drop(['Unnamed: 0'], axis=1)
    df.reset_index(drop=True, inplace=True)
    assert df.isnull().sum().sum() == 0, 'there is a NULL value in the loaded data'
    
    chip = 'R1005'
    input_vars = ['PRE_L','PRE_W']
    output_vars = ['POST_L','POST_W']
    vars = input_vars+output_vars
    
    xy = customMOM4chipsample(df, input_vars=vars, num_samples=100, chiptype=chip, random_state=42).rename(columns={'PRE_L': 0, 'PRE_W': 1, 'POST_L': 2, 'POST_W': 3})                
    y = xy.iloc[:, 2:].apply(np.linalg.norm, axis=1).astype(np.float32)  # objective: L-2 norm
    x1, x2 = torch.from_numpy(xy.iloc[:, 0].to_numpy()), torch.from_numpy(xy.iloc[:, 1].to_numpy())
    dist = f(x1, x2)
    
    
    
    # x1 = torch.ones(100, 1)*40
    # x2 = torch.ones(100, 1)*10
    # dist = f(x1.squeeze(-1), x2.squeeze(-1))
    # print('sample:', x1[1:2], x2[1:2], '-->', dist[1:2])
    print(x1.shape, x2.shape, dist.shape)
    