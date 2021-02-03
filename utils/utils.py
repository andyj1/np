#!/usr/bin/env python3

import gc
import time
import warnings

import joblib
import numpy as np
import torch
from botorch.exceptions import BadInitialCandidatesWarning
from gpytorch import settings
from torch.distributions.kl import kl_divergence
import pandas as pd

'''
given a list of tensors, return a list of cuda state booleans
'''
def checkParamIsSentToCuda(args):
    status = []
    for i, arg in enumerate(args):
        try:
            status.append(arg.is_cuda)
        except:
            status.append(False)
    return status

'''
computes Kullback-Leibler divergence between two Gaussian distributions p and q 
'''
def KLD_gaussian(p, q):
    '''
    computes KL(p||q)
    '''
    return kl_divergence(p, q).mean(dim=0).sum()

'''
loads reflow oven model from *reflow_oven/models*
'''
def loadReflowOven(args):
    # load RF regressor 
    loadRFRegressor_start = time.time()
    
    # reflow oven maps [PRE] to [POST]
    reflow_oven_model_path = args.load_rf
    regr_multirf = joblib.load(reflow_oven_model_path)

    loadRFRegressor_end = time.time()
    time_taken = loadRFRegressor_end - loadRFRegressor_start
    return regr_multirf, time_taken

'''
objective function value which the surrogate model is outputting and the acquisition function is minimizing
'''
def objective(x, y):
    return np.linalg.norm((x,y))

'''
simulates POST (x,y) given PRE (x,y)
'''
def reflow_oven(inputs, model):
    cuda_status = checkParamIsSentToCuda(inputs)
    if cuda_status == [True]:
        inputs = inputs.detach().cpu().numpy()
    
    # evaluate
    outputs = model.predict(inputs)
    
    return outputs

'''
set gpytorch's Cholesky decomposition computation type
(either exact if the matrx is Hermitian positive-definite, or approximate using low rank approximation using the Lanczos algorithm)
'''
def set_decomposition_type(cholesky: bool):
    '''arguments
    1. covar_root_decomposition: decomposition using low-rank approx using the Lanczos algorithm (False -> use Cholesky)
    
    2. log_prob: computed using a modified conjugate gradients algorithm (False -> use Cholesky)
    
    3. solves: computes positive-definite matrices with preconditioned conjugate gradients (False -> use Cholesky)
    '''
    set_bool = not cholesky
    if cholesky:
        settings.fast_computations(covar_root_decomposition=set_bool, 
                                    log_prob=set_bool, 
                                    solves=set_bool)

'''
set device, suppress warnings, set seed value
'''
def set_global_params():    
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Running on:',str(device).upper())

    # suppress runtime warnings
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # set seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    
    # sets behchmark mode in cudnn
    # benchmark mode is good whenever your input sizes for your network do not vary
    # ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # torch.backends.cudnn.benchmark = True

    return device

'''
garbage collect and empty cache memory in cuda device
'''
def clean_memory():    
    print('[INFO] garbage collect, torch emptying cache...')
    gc.collect()
    torch.cuda.empty_cache()

'''
make numpy array into pandas series
'''
def make_pd_series(nparray: np.array, name: str):
    return pd.Series(nparray, dtype=float, name=name)
