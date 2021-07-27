#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from toy import self_alignment


def f(x, y):
    # x, y: vector (each of which is sample size)
    x_grid_pre, y_grid_pre = torch.meshgrid(x, y)
    
    # self-alignment
    if len(x.shape) != 2:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
    xy = torch.cat((x, y), 1)
    xy_shifted, method = self_alignment.self_alignment(xy)
    x_shifted = xy_shifted[:, 0]
    y_shifted = xy_shifted[:, 1]
    
    # distance
    xy_to_L2distance = lambda inputs: torch.linalg.norm(inputs, dtype=torch.float64)
    
    # prepare grid space
    x_grid_post, y_grid_post = torch.meshgrid(x_shifted, y_shifted)
    distances = torch.zeros(x_grid_pre.shape, dtype=torch.float32)
    
    for i in range(x_grid_pre.shape[0]):
        for j in range(x_grid_pre.shape[1]):
            # append POST distance
            point = torch.FloatTensor([x_grid_post[i,j], y_grid_post[i,j]])
            distances[i,j] = torch.linalg.norm(point, dtype=torch.float64)
    
    # distance: objective over meshgrid space
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
        plt.title(f'{num_dim}-D {model_type.upper()}: minimize POST distance (iter: {iteration})')
    else:
        plt.title(f'{num_dim}-D {model_type.upper()}: minimize POST distance')
    return ax


if __name__ == '__main__':
    
    x1 = torch.ones(100, 1)*40
    x2 = torch.ones(100, 1)*10
    dist = f(x1.squeeze(-1), x2.squeeze(-1))
    # print('sample:', x1[1:2], x2[1:2], '-->', dist[1:2])
    print(x1.shape, x2.shape, dist.shape)