import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch

from torch.optim import SGD
import torch.optim as optim
from tqdm import trange
import sys

from NPModel import NP
from np_utils import log_likelihood, KLD_gaussian, random_split_context_target

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float
    
class SurrogateModel(object):
    def __init__(self, epochs=100):
        super(SurrogateModel, self).__init__()

        self.epochs = epochs
        # if not neural:
        self.model = None
        # else:
        #     self.model = NP(cfg['np']['hidden_dim'] , cfg['np']['decoder_dim'], cfg['np']['z_samples']).to(device)
        
        ''' alternative (default)
        self.model = SingleTaskGP(X, y)
        self.mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
        self.mll.to(dtype)
        # fit_gpytorch_model uses L-BFGS-B to fit the parameters by default
        fit_gpytorch_model(self.mll)
        '''
    # custom fitting
    def fitGP(self, train_X, train_Y):
        # initialize model
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        mll.to(train_X)
        mll.to(device)

        # default wrapper tor training
        # fit_gpytorch_model(mll)
        mll.train()
        mll, info_dict = fit_gpytorch_torch(mll)
        # mll.eval()
        
        # fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch)

        # the following code works, but has multiple lengthscale outputs (can't observe lengthscale)
        # model.train() # set train mode
        # optimizer = SGD([{'params': model.parameters()}], lr=0.1)        
        # t = trange(self.epochs, desc='', leave=False)
        # for epoch in t:
        #     t.set_description("[Train] Epoch %i / %i\t" % (epoch, self.epochs))

        #     optimizer.zero_grad()
        #     output = model(train_X)
        #     loss = -mll(output, model.train_targets)
        #     loss.backward()
        #     print(f"Epoch {epoch+1:>3}/{self.epochs} - Loss: {loss.item():>4.3f}"\
        #         f" - noise: {model.likelihood.noise.item():>4.3f}")
        #     # print(f"lengthscale: {model.covar_module.base_kernel.lengthscale:>4.3f}")
        #     optimizer.step()
            
        self.model = model.cpu()
    
    # TODO: 
    def fitNP(self, train_X, train_Y, cfg):        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # split into context and target
            x_context, y_context, x_target, y_target = random_split_context_target(train_X, train_Y, cfg['np']['num_context'])
            # send to gpu
            x_context = x_context.to(device)
            y_context = y_context.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)
            # forward pass
            mu, std, z_mean_all, z_std_all, z_mean, z_std = self.model(x_context, y_context, x_target, y_target)
            # loss calculation
            loss = -log_likelihood(mu, std, y_context) + KLD_gaussian(z_mean_all, z_std_all, z_mean, z_std)
            # backprop
            loss.backward()
            training_loss = loss.item()
            optimizer.step()
            print('epoch: {} loss: {}'.format(epoch, training_loss/200))
        
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
        
    # TODO:
    def evalNP(self, train_X, train_Y, ):
        mu, std = None, None
        # random z sample from normal of size (1, z_dim),
        # get mu, std from x_target and z_sample (xz to y)
        x_context, y_context, x_target, y_target = random_split_context_target(train_X, train_Y, cfg['np']['num_context'])
        # send to gpu
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)
        
        # forward pass
        mu, std, _, z_mean_all, z_std_all, z_mean, z_std = self.model(x_context, y_context, x_target, y_target)
        return mu, std
        