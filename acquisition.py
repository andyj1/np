#!/usr/bin/env python3
# original
# from botorch.acquisition.analytic import UpperConfidenceBound
# from botorch.acquisition.objective import ScalarizedObjective
# from botorch.optim.optimize import optimize_acqf

import torch
from custom.analytic_anp import UpperConfidenceBound
from custom.optimize import optimize_acqf_anp

class Acquisition(object):
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.device = device
        self.model = model
        
        # bounds
        self.dim = cfg['input_dim']
        self.bounds = torch.stack(
            [torch.ones(self.dim, device=self.device) * self.cfg['bounds'][0], 
             torch.ones(self.dim, device=self.device) * self.cfg['bounds'][1]])
        # self.bounds = torch.stack([torch.FloatTensor(self.cfg['bound1']), torch.FloatTensor(self.cfg['bound2'])]).to(self.device)
    
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
                                            # maximize=True)
        

    def optimize(self):
        with torch.autograd.set_detect_anomaly(False):
            candidate, acq_value = optimize_acqf_anp(self.acq_fcn,
                                                    bounds=self.bounds,
                                                    q=self.q,
                                                    num_restarts=self.num_restarts,
                                                    raw_samples=self.raw_samples)
            # candidate, acq_value = optimize_acqf(self.acq_fcn,
            #                                     bounds=self.bounds,
            #                                     q=self.q,
            #                                     num_restarts=self.num_restarts,
            #                                     raw_samples=self.raw_samples)
        return candidate, acq_value

    def optimize_acquisition(self, X):
        # X: valid x points
        # Xsamples: randomly sampled within domain (bounds)
        
        return
        
        