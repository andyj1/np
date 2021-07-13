import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets as ds
from torchvision import transforms as transforms
from yaml.error import YAMLError

import anp_utils
import bo_utils

from toy import toydata, self_alignment

# only consider 1-D in this case
class CustomData(torch.utils.data.Dataset):
    ''' options:
            parabola (N-Dimensional)
            sine (1-Dimensional)
    '''
    def __init__(self, input_dim, num_samples, type, cfg):
        super(CustomData, self).__init__()
        
        self.num_dim = input_dim
        self.num_samples = num_samples
        self.type = type
        self.cfg = cfg
        
        df = self.fetch_dataframe(type)
        self.x = df.iloc[:, 0:self.num_dim].values
        self.y = df.iloc[:, self.num_dim:self.num_dim+1].values
        
        # print(self.x.shape, self.y.shape)
    def __len__(self):
        return self.num_samples # len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def fetch_dataframe(self, type):
        assert type in ['parabola','sine', 'toy']
        
        if type == 'parabola':
            # self.num_dim = 1 or 2
            ''' 
                sphere: https://www.sfu.ca/~ssurjano/Code/spherefm.html 
                squared sum along all dimensions
            '''
            min_val, max_val = -10, 10
            
            # x: values in [min_val, max_val] in random order
            rng = np.random.default_rng(42)
            
            x = rng.random(size=(self.num_samples, self.num_dim)) * (max_val-min_val) + min_val
            x = pd.DataFrame(x, columns=[i+1 for i in range(self.num_dim)]).astype(np.float32)
            # y = x ** 2
            y = x.apply(lambda x: x**2).sum(axis=1).rename('sphere').astype(np.float32)
        
        elif type == 'toy':
            # self.num_dim = 2
            
            toy = toydata.ToyData()            
            # x = torch.cat([toy.preLW()], dim=1)
            x = toy.preLW()
            x, method = self_alignment.self_alignment(x)
            x = pd.DataFrame(x, columns=[i+1 for i in range(self.num_dim)]).astype(np.float32)
            # y = pd.DataFrame(y, columns=['y']).astype(np.float32)
            y = x.apply(np.linalg.norm, axis=1).astype(np.float32)  # objective: L-2 norm
            
        elif type == 'sine':
            self.num_dim = 1
            rng = np.random.default_rng()
            
            sin_amp1 = self.cfg['sin_amp1']
            sin_freq1 = self.cfg['sin_freq1']
            sin_phase1 = self.cfg['sin_phase1']
            
            sin_amp2 = self.cfg['sin_amp2']
            sin_freq2 = self.cfg['sin_freq2']
            sin_phase2 = self.cfg['sin_phase2']
            
            sin_noise = self.cfg['sin_noise']
            
            x = torch.linspace(-6, 6, self.num_samples)
            x = x.unsqueeze(1)
            sin1 = sin_amp1 * torch.sin(x * sin_freq1 + sin_phase1)
            sin2 = sin_amp2 * torch.sin(x * sin_freq2 + sin_phase2)
            sin_noise = torch.randn(x.size()) * math.sqrt(sin_noise)
            y = sin1 + sin2 + sin_noise
            
            # structure as pandas dataframe            
            x = pd.DataFrame(x, columns=['x']).astype(np.float32).sort_values(by='x', ascending=True)
            y = pd.DataFrame(y, columns=['y']).astype(np.float32)       

        df =  pd.concat((x, y), axis=1)
        # print('data shape:', df.shape)
        return df  # .reset_index(drop=True, inplace=True)
    
if __name__ == '__main__':
    print('='*10, 'toy data','='*10, )
    total_count = 1000
    context_count = 200
    datatype = 'toy'
    dataset = CustomData(input_dim=2, num_samples=total_count, type=datatype)
    bsz = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=bsz, shuffle=True, num_workers=4)
    sample_batch = next(iter(dataloader))
    x, y = sample_batch[0], sample_batch[1]
    print('[full x, y]:', x.shape, y.shape)
    
    
    print('='*10, 'sine data','='*10, )
    total_count = 1000
    context_count = 200
    datatype = 'sine'
    dataset = CustomData(input_dim=1, num_samples=total_count, type=datatype)
    bsz = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=bsz, shuffle=True, num_workers=4)
    sample_batch = next(iter(dataloader))
    x, y = sample_batch[0], sample_batch[1]
    print('[full x, y]:', x.shape, y.shape)
    
    '''
    context_idx, target_idx = utils.context_target_split(total_count, context_count)
    batch = 1
    x = x.expand(batch, -1, -1)
    y = y.expand(batch, -1, -1)
    x_context, y_context = x[:, context_idx, :], y[:,  context_idx, :]
    x_target, y_target = x[:, target_idx, :], y[:, target_idx, :]
    x_all = torch.cat([x_context, x_target], dim=1) # dim 1: num_samples
    y_all = torch.cat([y_context, y_target], dim=1) # dim 1: num_samples
    print('[split into context, target]:', x_context.shape, y_context.shape, x_target.shape, y_target.shape)
    
    utils.plot_functions(x_target, y_target, x_context, y_context, y_target, 0)
    
    order = torch.argsort(x, dim=1)
    x = x.squeeze()[order].squeeze()
    y = y.squeeze()[order].squeeze()
    
    order = torch.argsort(x_context, dim=1)
    x_context = x_context.squeeze()[order].squeeze()
    y_context = y_context.squeeze()[order].squeeze()
    
    # order = torch.argsort(x_target, dim=1)
    # x_target = x_target.squeeze()[order].squeeze()
    # y_target = y_target.squeeze()[order].squeeze()
    
    order = torch.argsort(x_all, dim=1)
    x_all = x_all.squeeze()[order].squeeze()
    y_all = y_all.squeeze()[order].squeeze()
    print('[target(all)]:', x_all.shape, y_all.shape)
    
    plt.plot(x.squeeze(), y.squeeze(), 'k--', linewidth=0.5, label='all')
    # plt.plot(x_target, y_target, 'b.', markersize=1, label='target')
    plt.plot(x_all, y_all, 'b.', markersize=3, linewidth=1, label='target')
    plt.plot(x_context, y_context, 'go', markersize=6, linewidth=1, label='context')
    plt.legend()
    plt.show()
    '''
    
    # print('='*10, 'parabola data','='*10, )
    # dataset = CustomData(input_dim=1, num_samples=5, type='parabola')
    # bsz = len(dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True, num_workers=4)
    # sample_batch = next(iter(dataloader))
    # print(sample_batch[0].shape, sample_batch[1].shape)

    
    # print('='*10, 'MOM4 data','='*10, )
    # dataset = MOM4Data(input_dim=1, num_samples=5)
    # bsz = len(dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True, num_workers=4)
    # sample_batch = next(iter(dataloader))
    # print(sample_batch[0].shape, sample_batch[1].shape)
    