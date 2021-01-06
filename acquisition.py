import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
# from botorch.acquisition.acquisition import AcquisitionFunction
from custom.analytic import (ExpectedImprovement, UpperConfidenceBound)
# from custom.analytic_np import (ExpectedImprovement, UpperConfidenceBound) # NP version
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler

from custom.optimize import optimize_acqf

class Acquisition(object):
    def __init__(self, cfg, model, device, beta=None, best_f=None):
        option = cfg['acquisition']['option']
        
        self.beta = 0 if beta is None else cfg['acquisition']['beta']
        self.best_f = 0 if best_f is None else cfg['acquisition']['best_f']
        
        dim = 2
        # bounds for each column of X
        bound_limit = cfg['acquisition']['bounds']
        self.bounds = torch.stack([-torch.ones(dim, device=device) * bound_limit, torch.ones(dim, device=device) * bound_limit])
        self.q = cfg['acquisition']['q']
        self.num_restarts = cfg['acquisition']['num_restarts']
        self.raw_samples = cfg['acquisition']['raw_samples']
        self.candidate = None
        self.acq_value = None
        
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
        
    # for GP
    def optimize(self):
        # if sequential is kept turned off, it performs the following
        # 1. generate initial candidates
        # 2. get candidates from scipy.optimize.minimize or torch.optim
        candidate, acq_value = optimize_acqf(self.acq_fcn, 
                                            bounds=self.bounds, 
                                            q=self.q, 
                                            num_restarts=self.num_restarts, 
                                            raw_samples=self.raw_samples)
        return candidate, acq_value

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
