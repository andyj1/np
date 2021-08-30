import configparser
import gc
import warnings

import torch

import reflow_soldering

cfg = configparser.ConfigParser()
cfg.read('config.ini')

stripped = cfg.items('train')[-1][1].strip('[],')
mounter_noise_min, mounter_noise_max = int(stripped[0]), int(stripped[3:])

# mounter noise: uniformly distributed ~U(a, b)
scaler = lambda x, a, b: b + (a - b) * x

def setup():
    torch.cuda.empty_cache()
    gc.collect()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def black_box_function(x1, x2): 
    '''objective function
    Note:
        match this function with f(x,y) in *visualize_objective_space.py*
    
    '''
    
    # add self-alignment to shift PRE to POST
    shifted_xs, method = reflow_soldering.self_alignment(x1, x2)
    x1, x2 = torch.chunk(shifted_xs, chunks=2, dim=-1)
    
    # add mounter noise
    # x1 += scaler(torch.rand(x1.shape), mounter_noise_min, mounter_noise_max)
    # x2 += scaler(torch.rand(x1.shape), mounter_noise_min, mounter_noise_max)
    
    norm_distance = torch.zeros(x1.shape)
    for i in range(norm_distance.shape[0]):
        point = torch.FloatTensor([x1[i], x2[i]])
        norm_distance[i] = torch.linalg.norm(point, dtype=torch.float64)
        # print(point, norm_distance[i])
    # at this point, utility (acq func) only maximizes,
    # so negate this to minimize
    
    # distance: objective over sample space
    # return negative since BO framework maximizes the objective
    return -norm_distance

if __name__ == '__main__':
    
    x1 = torch.ones(100, 1)*40
    x2 = torch.ones(100, 1)*10
    dist = black_box_function(x1, x2)
    print(x1.shape, x2.shape, dist.shape)
    print('sample:', x1[1:2], x2[1:2], '-->', dist[1:2])