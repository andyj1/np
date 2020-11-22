import torch
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import UpperConfidenceBound

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
        
    def optimize(self):
        self.candidate, self.acq_value = optimize_acqf(self.acq_fcn, 
                                                        bounds=self.bounds, 
                                                        q=self.q, 
                                                        num_restarts=self.num_restarts, 
                                                        raw_samples=self.raw_samples)
        print('\ncandidate:',self.candidate,
              'acq_value:',self.acq_value)
        return self.candidate, self.acq_value
