#!/usr/bin/env python3
import torch
# from botorch.acquisition.analytic import UpperConfidenceBound
# from botorch.acquisition.objective import ScalarizedObjective

from custom.analytic_anp import (UpperConfidenceBound)
from custom.optimize import optimize_acqf_anp
# from botorch.optim.optimize import optimize_acqf

class Acquisition(object):
    def __init__(self, cfg, model, device):
        self.device = device
        
        # bounds
        self.dim = cfg['input_dim']
        self.bounds = cfg['bounds']
        self.bounds = torch.stack(
            [-torch.ones(self.dim, device=self.device) * self.bounds[0], 
             torch.ones(self.dim, device=self.device) * self.bounds[1]])

        # UCB parameters
        self.beta = cfg['beta']
        self.q = cfg['q']
        self.num_restarts = cfg['num_restarts']
        self.raw_samples = cfg['raw_samples']
        
        # acquisition function type
        self.acq_fcn = UpperConfidenceBound(model, 
                                            beta=self.beta, 
                                            objective=None,
                                            maximize=False)
        

    def optimize(self):
        with torch.autograd.set_detect_anomaly(False):
        
            candidate, acq_value = optimize_acqf_anp(self.acq_fcn,
                                                bounds=self.bounds,
                                                q=self.q,
                                                num_restarts=self.num_restarts,
                                                raw_samples=self.raw_samples)
        return candidate, acq_value
