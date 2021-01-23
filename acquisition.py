#!/usr/bin/env python3


import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
# from botorch.acquisition.acquisition import AcquisitionFunction
# from custom.analytic import (ExpectedImprovement, UpperConfidenceBound)
from custom.analytic_np import (UpperConfidenceBound)
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler

from custom.optimize import optimize_acqf, optimize_acqf_NP

class Acquisition(object):
    def __init__(self, cfg, model, device, beta=None, best_f=None):
        cfg_acq = cfg['acquisition']
        
        option = cfg_acq['option']
        self.beta = 0 if beta is None else cfg_acq['beta']
        self.best_f = 0 if best_f is None else cfg_acq['best_f']
        self.q = cfg_acq['q']
        self.num_restarts = cfg_acq['num_restarts']
        self.raw_samples = cfg_acq['raw_samples'] # init: same as number of target samples (train_num_samples - np_num_context); need to adjust this at every iteration
        self.candidate = None
        self.acq_value = None
        
        # bounds for each column of input
        dim = len(cfg['MOM4']['input_var'])
        bound_limit = cfg_acq['bounds']
        self.bounds = torch.stack([-torch.ones(dim, device=device) * bound_limit, torch.ones(dim, device=device) * bound_limit])
        
        if option == 1:
            # 1. UCF
            self.acq_fcn = UpperConfidenceBound(model, beta=self.beta, maximize=False)
        elif option == 2:
            # 2. EI
            self.acq_fcn = ExpectedImprovement(model, best_f=self.best_f, maximize=False)
        elif option == 3:
            # 3. Sobol Quasi-Monte Carlo Normal sampler-based qEI 
            sampler = SobolQMCNormalSampler(num_samples=100, seed=0, resample=False)
            self.acq_fcn = qExpectedImprovement(
                model, best_f=self.best_f, sampler=sampler, maximize=False
            )
        elif option == 4:
            # 4. IID Normal sampler-based qEI 
            sampler = IIDNormalSampler(num_samples=100, resample=True)     
            self.acq_fcn = qExpectedImprovement(
                # set best_f to train_Y.max()
                model, best_f=self.best_f, sampler=sampler, maximize=False
            )

    def optimize(self, np=False):
        # for GP
        if np == False:
            # if sequential is kept turned off, it performs the following
            # 1. generate initial candidates
            # 2. get candidates from scipy.optimize.minimize or torch.optim
            candidate, acq_value = optimize_acqf(self.acq_fcn, 
                                                bounds=self.bounds, 
                                                q=self.q, 
                                                num_restarts=self.num_restarts, 
                                                raw_samples=self.raw_samples)
        # for NP
        else:
            print('raw samples:', self.raw_samples)
            candidate, acq_value = optimize_acqf_NP(self.acq_fcn, 
                                            bounds=self.bounds, 
                                            q=self.q, 
                                            num_restarts=self.num_restarts, 
                                            raw_samples=self.raw_samples)
        return candidate, acq_value



