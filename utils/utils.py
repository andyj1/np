#!/usr/bin/env python3

import gc
import time
import warnings
import argparse
import joblib
import numpy as np
#!/usr/bin/env python3

import torch
from botorch.exceptions import BadInitialCandidatesWarning
from gpytorch import settings
from torch.distributions.kl import kl_divergence
import pandas as pd

def objective(outputs): # -> torch.FloatTensor
    '''
    objective function value which the surrogate model is outputting and the acquisition function is minimizing
    '''
    obj = lambda x, y: torch.norm(torch.FloatTensor([x, y])) # x,y are POST L, W (single sample)
    
    targets = torch.FloatTensor([obj(x,y) for x, y in outputs]).unsqueeze(1)
    targets = targets.to(outputs.device)
    return targets

def loadReflowOven(args):
    '''
    loads reflow oven model from *reflow_oven/models*
    '''
    # load RF regressor 
    loadRFRegressor_start = time.time()
    
    # reflow oven maps [PRE] to [POST]
    reflow_oven_model_path = args.load_rf
    regr_multirf = joblib.load(reflow_oven_model_path)

    loadRFRegressor_end = time.time()
    time_taken = loadRFRegressor_end - loadRFRegressor_start
    return regr_multirf, time_taken

def checkParamIsSentToCuda(args):
    '''
    given a list of tensors, return a list of cuda state booleans
    '''
    status = []
    for i, arg in enumerate(args):
        try:
            status.append(arg.is_cuda)
        except:
            status.append(False)
    return status

def KLD_gaussian(p, q):
    '''
    computes Kullback-Leibler divergence from Gaussian distribution p to q 
    '''
    return kl_divergence(p, q).mean(dim=0).sum()

def switchOrient(x90, y90):
    # switch 90 data to 0 data
    y0 = float(x90)
    x0 = float(-y90)
    return x0, y0 

def set_decomposition_type(cholesky: bool):
    '''
    set gpytorch's Cholesky decomposition computation type
    (either exact if the matrx is Hermitian positive-definite, or approximate using low rank approximation using the Lanczos algorithm)
    '''

    '''arguments
    1. covar_root_decomposition: decomposition using low-rank approx using the Lanczos algorithm (False -> use Cholesky)    
    2. log_prob: computed using a modified conjugate gradients algorithm (False -> use Cholesky)
    3. solves: computes positive-definite matrices with preconditioned conjugate gradients (False -> use Cholesky)
    '''
    if cholesky:
        settings.fast_computations(covar_root_decomposition=not cholesky, 
                                    log_prob=not cholesky, 
                                    solves=not cholesky)

def supress_warnings():    
    '''
    set device, suppress warnings, set seed value
    '''
    # suppress runtime warnings
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
def set_torch_seed(seed=42, benchmark=False):
    # set seed for reproducibility
    torch.manual_seed(seed)
    
    # sets behchmark mode in cudnn
    # benchmark mode is good whenever your input sizes for your network do not vary
    # ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = benchmark

def clean_memory():    
    '''
    garbage collect and empty cache memory in cuda device
    '''
    print('[INFO] garbage collect, emptying cache...')
    gc.collect()
    torch.cuda.empty_cache()

def make_pd_series(nparray: np.array, name: str):
    '''
    make numpy array into pandas series
    '''
    return pd.Series(nparray, dtype=float, name=name)
