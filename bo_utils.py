import configparser
import gc
import warnings

import numpy as np
import torch

from toy import self_alignment

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
    # add self-alignment to shift PRE to POST
    shifted_xs, method = self_alignment.self_alignment(x1, x2)
    x1, x2 = torch.chunk(shifted_xs, chunks=2, dim=1)
    
    # add mounter noise
    x1 += scaler(torch.rand(x1.shape), mounter_noise_min, mounter_noise_max)
    x2 += scaler(torch.rand(x1.shape), mounter_noise_min, mounter_noise_max)
    
    norm_distance = np.linalg.norm((x1, x2))
    # at this point, utility (acq func) only maximizes,
    # so negate this to minimize
    return -norm_distance
