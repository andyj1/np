import numpy as np
import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf


class AcquisitionFunction(object):
    def __init__(self, model, beta):
        self.beta = beta
        self.acq_fcn = UpperConfidenceBound(model, beta=beta)
        dim = 2
        self.bounds = torch.stack([-torch.ones(dim)*120, torch.ones(dim)*120])
        self.q = 1
        self.num_restarts = 10
        self.raw_samples = 40
        self.candidate = None
        self.acq_value = None
    
    # for GP
    def optimize(self):
        self.candidate, self.acq_value = optimize_acqf(self.acq_fcn, 
                                                        bounds=self.bounds, 
                                                        q=self.q, 
                                                        num_restarts=self.num_restarts, 
                                                        raw_samples=self.raw_samples)
        print('\ncandidate:',self.candidate, 'acq_value:',self.acq_value)
        return self.candidate, self.acq_value

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
