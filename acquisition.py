import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.analytic import (ExpectedImprovement,
                                          UpperConfidenceBound)
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler


class AcquisitionFunction(object):
    def __init__(self, cfg, model):
        option = cfg['acquisition']['option']
        if option == 1:
            # 1. UCF
            self.beta = cfg['acquisition']['beta']
            self.acq_fcn = UpperConfidenceBound(model, beta=self.beta)
        elif option == 2:
            # 2. EI
            self.best_f = cfg['acquisition']['best_f']
            self.acq_fcn = ExpectedImprovement(model, best_f=self.best_f)
        elif option == 3:
            # 3. MC-based qEI 
            self.best_f = cfg['acquisition']['best_f']
            sampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=False)        
            self.acq_fcn = qExpectedImprovement(
                model, best_f=self.best_f, sampler=sampler
            )
        
        dim = 2
        bound_limit = cfg['acquisition']['bounds']
        self.bounds = torch.stack([-torch.ones(dim) * bound_limit, torch.ones(dim) * bound_limit])
        self.q = cfg['acquisition']['q']
        self.num_restarts = cfg['acquisition']['num_restarts']
        self.raw_samples = cfg['acquisition']['raw_samples']
        self.candidate = None
        self.acq_value = None
    
    # for GP
    def optimize(self):
        return optimize_acqf(self.acq_fcn, 
                            bounds=self.bounds, 
                            q=self.q, 
                            num_restarts=self.num_restarts, 
                            raw_samples=self.raw_samples)

    # for NP
    # probability of improvement acquisition function
    def acquisition(X, Xsamples, model):
        # calculate the best surrogate score found so far
        yhat, _ = model(X) # forward pass prediction
        best = max(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = surrogate(model, Xsamples)
        mu = mu[:, 0]
        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std+1E-9))
        return probs

    # optimize the acquisition function
    def opt_acquisition(X, y, model):
        # random search, generate random samples
        Xsamples = random(100)
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        # calculate the acquisition function for each sample
        scores = acquisition(X, Xsamples, model)
        # locate the index of the largest scores
        ix = np.argmax(scores)
        return Xsamples[ix, 0]
