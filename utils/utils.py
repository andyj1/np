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


def reflow_oven(inputs, model):
    '''
    simulates POST (x,y) given PRE (x,y)
    '''
    cuda_status = checkParamIsSentToCuda(inputs)
    if cuda_status == [True]:
        inputs = inputs.detach().cpu().numpy()
    
    # evaluate
    outputs = model.predict(inputs)
    
    return outputs

def test_self_alignment(inputs, chip=None, chipname='R0402', option=1):
    '''
    self alignment simulation for toy data

    options
        1. random (constant) angular and translational shift
        2. weighted shift by the relative distance between chip (Pre-AOI) and SPI in x,y directions
        3. weighted shift by the amount of rotation and translation of the chip (Pre-AOI) + volume..? + rotation angle (again..?) (post theta calculation...?)
    
    inputs
        [pre_x, pre_y, pre_theta, spi_x1, spi_y1, spi_x2, spi_y2, volume_1, volume_2, volume_difference]
    
    outputs
        [post_chip, post_theta]
    '''
    if inputs.is_cuda: inputs = inputs.cpu()
    if option == 1:
        '''
        constant shift variables:
            angle (deg): randn(mean=0, std=10)
            position: randn(mean=40, std=20)
            
            total shift (dx, dy) = position (x,y) * (x,y) components of the angle vector
            
        return
            [original (x,y) + shift (dx, dy), theta(all zeros)]
        '''
        direction = 0.
        d_noise = 10.
        theta_deg = torch.normal(mean=direction, std=d_noise, size=(inputs.shape[0], 1))
        theta_rad = torch.deg2rad(theta_deg)

        distance = 40.
        noise = 20.
        position_shift = torch.normal(mean=distance, std=noise, size=(inputs.shape[0], 2))
        position_shift *= torch.cat([torch.cos(theta_rad), torch.sin(theta_rad)], dim=1)

        post_chip = inputs[:, 0:2] + position_shift
        post_theta = torch.zeros(size=(post_chip.shape[0], 1))
        outputs = torch.cat([post_chip, post_theta], dim=1)
        
    elif option == 2:
        '''
        simple self-alignment effect
            :demonstrates chip movement towards solder paste in the reflow oven
            :adds solder paste volume difference aspect (by multiplying to relative position difference in x)
            :adds resistance aspect for each chip
            :adds alpha for some additional variables not accounted for
        
        return  (= estimated chip position inspected at Post-AOI stage)
            [original (x,y) + shift (dx, dy), theta(all zeros)]
        '''
        # tension level by chip type
        tension_level = {'R0402': (1000*500)/(400*200), 'R0603': (1000*500)/(600*300), 'R1005': (1000*500)/(1000*500)}
        alpha = 0.1
        
        relative_spi = inputs[:, np.r_[3, 4]] - inputs[:, np.r_[0, 1]]
        
        # weigh relative spi
        weighted_relative_spi = relative_spi * torch.cat([(1 - inputs[:, -1]).unsqueeze(1), torch.ones(size=(inputs.shape[0], 1))], dim=1)
        weighted_relative_spi *= tension_level[chipname]
        weighted_relative_spi *= alpha

        # outs = []
        # for datum in inputs:
        #     pre_x, pre_y, _, SPI_x, SPI_y, _, _, _, _, SPI_diff = datum

        #     rel_SPI_x = (SPI_x - pre_x) * (1 - SPI_diff)
        #     rel_SPI_y = SPI_y - pre_y
                
        #     post_x = pre_x + rel_SPI_x * tension_level[chipname] * alpha
        #     post_y = pre_y + rel_SPI_y * tension_level[chipname] * alpha
        #     outs.append([post_x, post_y])
        # post_chip = torch.FloatTensor(np.array(outs))

        post_chip = inputs[:, 0:2] + weighted_relative_spi
        post_theta = torch.zeros(size=(post_chip.shape[0], 1))
        outputs = torch.cat([post_chip, post_theta], dim=1)

    elif option == 3:
        # [pre_x, pre_y, pre_theta, spi_x1, spi_y1, spi_x2, spi_y2, volume_1, volume_2, volume_difference]
        # pre_theta_rad = torch.deg2rad(inputs[:, 2]) # pre angle
    
        relative_spi = inputs[:, np.r_[3,4,5,6]] - inputs[:, np.r_[0,1,0,1]]

        direction = 0.
        d_noise = 10.
        theta_deg = torch.normal(mean=direction, std=d_noise, size=(inputs.shape[0], 1))
        theta_rad = torch.deg2rad(theta_deg)

        tension_level = {'R0402': 50/8, 'R0603': 50/18, 'R1005': 1}
        
        alpha = 0.2
        beta = 0.0005
        post_chip = []
        post_theta = []
        for i in range(inputs.shape[0]):
            val = torch.FloatTensor([[relative_spi[i, 0], relative_spi[i, 2]],
                                    [relative_spi[i, 1], relative_spi[i, 3]],
                                    [1,1]])
            rotation_matrix = torch.FloatTensor([[torch.cos(theta_rad[i]), -torch.sin(theta_rad[i]), 0],
                                                [torch.sin(theta_rad[i]), torch.cos(theta_rad[i]), 0],
                                                [0, 0, 1]])
            translation_matrix = torch.FloatTensor([[-chip['length']/2, chip['length']/2],
                                                    [-chip['width']/2, chip['width']/2],
                                                    [0, 0]])
            val = rotation_matrix @ val - translation_matrix # 3x3 * 3x2 - 3x2 = 3x2
            
            volume_matrix = torch.FloatTensor([[-inputs[i, 7], inputs[i, 7]],
                                              [-inputs[i, 8], inputs[i, 8]]])
            val = val @ volume_matrix # 3x2 * 2x2 = 3x2
            
            tension = torch.sum(val, dim=0)[0:2] * tension_level[chipname] # 1x2

            post_x = inputs[i, 0] + alpha * tension[0] * torch.cos(theta_rad[i])
            post_y = inputs[i, 1] + alpha * tension[0] * torch.sin(theta_rad[i])
            post_chip.append([post_x, post_y])
            post_theta.append(inputs[i, 2] + torch.rad2deg(beta * tension[1]))
        post_chip = torch.FloatTensor(post_chip)
        post_theta = torch.FloatTensor(post_theta).unsqueeze(-1)
        outputs = torch.cat([post_chip, post_theta], dim=1)
    return outputs


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
    return np.linalg.norm((x,y))


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
