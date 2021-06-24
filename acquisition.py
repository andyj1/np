#!/usr/bin/env python3
'''
    current acquisition functions support only q=1 (single-outcome case)
    for multi-objective, there are multi-objective acquisition functions available,
    given the surrogate model can output multi-outcome (for multi-output objective)
'''
import torch
from custom.optimize import optimize_acqf, optimize_acqf_NP
# from custom.optimize import optimize_acqf_NP
# from botorch.optim import optimize_acqf
class Acquisition(object):
    def __init__(self, cfg, model, device, beta=None, best_f=None, model_type='GP'):
        cfg_acq = cfg['acquisition']

        option = cfg_acq['option']
        self.beta = 0 if beta is None else cfg_acq['beta']
        self.best_f = 0 if best_f is None else cfg_acq['best_f']
        self.q = cfg_acq['q']
        self.num_restarts = cfg_acq['num_restarts']
        self.raw_samples = cfg_acq['raw_samples']
        self.candidate = None
        self.acq_value = None
        dim = len(cfg['MOM4']['input_var'])
        bound_limit = cfg_acq['bounds']
        self.bounds = torch.stack(
            [-torch.ones(dim, device=device) * bound_limit, torch.ones(dim, device=device) * bound_limit])

        if 'ANP' in model_type:
            from custom.analytic_anp import (UpperConfidenceBound)
        elif 'NP' in model_type:
            from custom.analytic_np import (UpperConfidenceBound, ExpectedImprovement)
        elif 'GP' in model_type:
            from custom.analytic_gp import (ExpectedImprovement, UpperConfidenceBound)

        # acquisition function type
        if option == 1:
            self.acq_fcn = UpperConfidenceBound(model, beta=self.beta, maximize=False)
        elif option == 2:
            self.acq_fcn = ExpectedImprovement(model, best_f=self.best_f, maximize=False)

    def optimize(self, np=False):
        with torch.autograd.set_detect_anomaly(False):
            # for GP
            if np == False:
                # if sequential is kept turned off, it performs the following
                # 1. generate initial candidates
                # 2. get candidates from scipy.optimize.minimize
                candidate, acq_value = optimize_acqf(self.acq_fcn,
                                                    bounds=self.bounds,
                                                    q=self.q,
                                                    num_restarts=self.num_restarts,
                                                    raw_samples=self.raw_samples)
            # for NP
            else:
                candidate, acq_value = optimize_acqf_NP(self.acq_fcn,
                                                        bounds=self.bounds,
                                                        q=self.q,
                                                        num_restarts=self.num_restarts,
                                                        raw_samples=self.raw_samples)
                
            # print(f'[INFO] \033[91m raw samples: {self.raw_samples}, num_restarts: {self.num_restarts} \033[0m')
        return candidate, acq_value
