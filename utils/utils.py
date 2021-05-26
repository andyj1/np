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

def parse():
    parse_start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', help='sets to toy dataset')
    parser.add_argument('--load_rf', default='reflow_oven/models_2var/regressor_R1005_50_trees_100_deep_random_forest.pkl', type=str, help='path to reflow oven model')
    parser.add_argument('--model', default='GP', type=str, help='surrogate model type')
    parser.add_argument('--load', default=None, type=str, help='path to checkpoint [pt] file')
    parser.add_argument('--chip', default=None, type=str, help='chip part type')
    # parser.add_argument('--not_increment_context', default=True, action='store_false', help='increments context size over iterations, instead of target size')
    parser.add_argument('--cholesky', default=False, action='store_true', help='sets boolean to use cholesky decomposition')
    args = parser.parse_args()
    
    parse_end = time.time(); 
    print('[INFO] parsing argumentstook: %.3f seconds' % (parse_end - parse_start))
    return args

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
    computes Kullback-Leibler divergence between two Gaussian distributions p and q 
    '''
    '''
    computes KL(p||q)
    '''
    return kl_divergence(p, q).mean(dim=0).sum()

def switchOrient(x90, y90):
    # switch 90 data to 0 data
    y0 = float(x90)
    x0 = float(-y90)
    return x0, y0 

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

def objective(x, y):
    '''
    objective function value which the surrogate model is outputting and the acquisition function is minimizing
    '''
    return np.array([objective(x,y) for x, y in zip(x[:], y[:])])

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
    set_bool = not cholesky
    if cholesky:
        settings.fast_computations(covar_root_decomposition=set_bool, 
                                    log_prob=set_bool, 
                                    solves=set_bool)

def set_global_params():    
    '''
    set device, suppress warnings, set seed value
    '''
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); 

    # suppress runtime warnings
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # set seed for reproducibility
    # SEED = 42
    # torch.manual_seed(SEED)
    
    # sets behchmark mode in cudnn
    # benchmark mode is good whenever your input sizes for your network do not vary
    # ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # torch.backends.cudnn.benchmark = True

    return device

def clean_memory():    
    '''
    garbage collect and empty cache memory in cuda device
    '''
    print('[INFO] garbage collect, torch emptying cache...')
    gc.collect()
    torch.cuda.empty_cache()

def make_pd_series(nparray: np.array, name: str):
    '''
    make numpy array into pandas series
    '''
    return pd.Series(nparray, dtype=float, name=name)
