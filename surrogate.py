import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from torch.optim import SGD
import torch.optim as optim
from tqdm import trange
import sys

from np_model import NP
from np_utils import log_likelihood, KLD_gaussian, random_split_context_target

''' class for surrogate model
Methods:
    1) fit/train 
        input: vector of X's (multi-dim), vector of Y (single-dim)
        ->
        output: trained model
    2) forward/predict
        input: sample point x
        ->
        output: mean and variance at x
'''

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float
    
class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()

        self.epochs = 10
        self.model = None
        
        ''' alternative (default)
        self.model = SingleTaskGP(X, y)
        self.mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
        self.mll.to(dtype)
        # fit_gpytorch_model uses L-BFGS-B to fit the parameters by default
        fit_gpytorch_model(self.mll)
        '''
    # custom fitting
    def fit(self, train_X, train_Y):
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        mll.to(dtype)
        fit_gpytorch_model(mll)
        '''
        model.train() # set train mode
        optimizer = SGD([{'params': model.parameters()}], lr=0.1)        
        t = trange(self.epochs, desc='', leave=False)
        for epoch in t:
            optimizer.zero_grad()
            output = model(train_X)
            loss = -mll(output, model.train_targets)
            loss.backward()
            # print(f"Epoch {epoch+1:>3}/{self.epochs} - Loss: {loss.item():>4.3f} "
                #   f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():>4.3f} " f"noise: {model.likelihood.noise.item():>4.3f}")
            optimizer.step()
            
            # t.set_description("[Train] Epoch %i / %i\t" % (epoch, self.epochs))
            # t.refresh()
        '''
        self.model = model
    
    # TODO: modify this
    def fitNP(self, train_X, train_Y, cfg):        
        model = NP(cfg['hidden_dim'] , cfg['decoder_dim'], cfg['z_samples']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # split into 
            x_context, y_context, x_target, y_target = random_split_context_target(train_X, train_Y, cfg['num_context'])
            # send to gpu
            x_context = x_context.to(device)
            y_context = y_context.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)
            # forward pass
            mu, std, z_mean_all, z_std_all, z_mean, z_std = model(x_context, y_context, x_target, y_target)
            # loss calculation
            loss = -log_likelihood(mu, std, y_context) + KLD_gaussian(z_mean_all, z_std_all, z_mean, z_std)
            # backprop
            loss.backward()
            training_loss = loss.item()
            optimizer.step()
            print('epoch: {} loss: {}'.format(epoch, training_loss/200))
        self.model = model
        
    '''
    eval: performs evaluation at test points and return mean and lower, upper bounds
    '''
    def eval(self, test_X):
        posterior, variance = None, None
        if self.model is not None:
            self.model.eval() # set eval mode
            with torch.no_grad():
                posterior = self.model.posterior(test_X)
                # upper and lower confidence bounds (2 standard deviations from the mean)
                lower, upper = posterior.mvn.confidence_region()
            return posterior.mean.cpu().numpy(), (lower.cpu().numpy(), upper.cpu().numpy())