#!/usr/bin/env python3

import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from custom.fit import fit_gpytorch_torch

from torch.optim import SGD
import torch.optim as optim
from tqdm import trange
import sys

import random
import numpy as np

from NPModel import NP
from utils.np_utils import log_likelihood, KLD_gaussian, random_split_context_target

class SurrogateModel(object):
    def __init__(self, train_X, train_Y, args, cfg, writer, device=torch.device('cpu'), epochs=100):
        super(SurrogateModel, self).__init__()
        
        self.epochs = epochs
        
        # intiialize model for nominal purposes only
        if not args.np :
            self.model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            self.model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
            mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
            mll.to(train_X)
        else:
            self.model = NP(cfg['np']).to(device)
        
        # optimizer_cls = optim.AdamW
        # optimizer_cls = optim.SparseAdam # doesn't support dense gradients
        # self.optimizer_cls = optim.Adamax
        self.optimizer_cls = optim.Adam
        self.optimizer = self.optimizer_cls(self.model.parameters())
        
        # use GPU if available
        self.device = device
        self.dtype = torch.float
        self.writer = writer

    # custom GP fitting
    def fitGP(self, train_X, train_Y, cfg, toy_bool=False, epoch=0):
        chip = cfg['MOM4']['parttype']
        # re-initialize model
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        model.to(self.device)
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        mll.to(train_X)     # set to default data type
        mll.to(self.device) # upload to GPU

        # default wrapper tor training
        # fit_gpytorch_model(mll)
        mll.train()
        
        # customize optimizer in 'fit.py' in fit_gpytorch_torch()
        # optimizer need not have a closure
        optimizer_options = {'lr': 0.05, "maxiter": self.epochs, "disp": True} # for botorch
        
        ''' define custom optimizer using optimizer class: "self.optimizer_cls" '''
        # self.optimizer = self.optimizer_cls(model.parameters())
        self.optimizer = None # if None, defines a new optimizer within fit_gpytorch_torch
        
        mll, info_dict, self.optimizer = fit_gpytorch_torch(mll=mll, \
                                            optimizer_cls=self.optimizer_cls, \
                                            options=optimizer_options, \
                                            approx_mll=True, \
                                            custom_optimizer=self.optimizer, \
                                            device=self.device)
        loss = info_dict['fopt']
        self.model = mll.model
        
        # alternative to fit_gpytorch_torch; more general fit API
        # fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch)

        ''' uncomment the following code for custom fit  * but has multiple lengthscale outputs (can't observe lengthscale) '''
        # optimizer_options = {'lr': 0.05} # for pytorch
        # model.train() # set train mode
        # self.optimizer = self.optimizer_cls([{'params': model.parameters()}], **optimizer_options)        
        # DISPLAY_FOR_EVERY = self.epochs
        # t = trange(self.epochs, desc='', leave=False)
        # for train_epoch in t:

        #     self.optimizer.zero_grad()
        #     output = model(train_X)
        #     loss = -mll(output, model.train_targets)
        #     loss.backward()
        #     # t.set_description(f"[Train] Iter {train_epoch+1:>3}/{self.epochs} - Loss: {loss.item():>4.3f} - noise: {model.likelihood.noise.item():>4.3f}\n",refresh=False)
            
        #     # if epoch % DISPLAY_FOR_EVERY == 0:
        #     #     print(f"[Train] Iter {epoch+1:>3}/{self.epochs} - Loss: {loss.item():>4.3f} - noise: {model.likelihood.noise.item():>4.3f}")
        #     # print(f"lengthscale: {model.covar_module.base_kernel.lengthscale:>4.3f}")
        #     self.optimizer.step()
        
        checkpoint = {'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}
        if toy_bool:
            torch.save(checkpoint, f"ckpts/toy/checkpoint_{epoch}.pt")
        else:
            torch.save(checkpoint, f"ckpts/{chip}/checkpoint_{epoch}.pt")
    
    # TODO:
    def fitNP(self, train_X, train_Y, cfg, toy_bool=False, epoch=0):
        chip = cfg['MOM4']['parttype']
        info_dict = {}
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # split into context and target
            # x_context, y_context, x_target, y_target = random_split_context_target(train_X, train_Y, cfg['np']['num_context'])
            context_idx = random.sample(range(train_X.shape[0]), cfg['np']['num_context'])
            target_idx = np.delete(range(train_X.shape[0]), context_idx)
            x_context, x_target = train_X[context_idx], train_X[target_idx]
            y_context, y_target = train_Y[context_idx], train_Y[target_idx]

            # send to gpu
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)

            # forward pass
            # mu, std, z_mean_all, z_std_all, z_mean, z_std = self.model(x_context, y_context, x_target, y_target)
            X = [x_context, y_context, x_target]
            # mu, std, q_target, q_context = self.model(x_context, y_context, x_target, y_target)
            mu, std, q_target, q_context = self.model(X, y_target)

            # loss calculation
            # loss = -log_likelihood(mu, std, y_target) + KLD_gaussian(z_mean_all, z_std_all, z_mean, z_std)
            loss = -log_likelihood(mu, std, y_target) + KLD_gaussian(q_target, q_context)

            self.writer.add_scalar(f"Loss/train_NP_{cfg['train']['num_samples']}_samples_fitNP", loss.item(), epoch)
            # backprop
            loss.backward()
            training_loss = loss.item()
            optimizer.step()
            print('epoch: {} loss: {}'.format(epoch, training_loss))

        checkpoint = {'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}
        if toy_bool:
            torch.save(checkpoint, f"ckpts/toy/checkpoint_{epoch}_NP.pt")
        else:
            torch.save(checkpoint, f"ckpts/{chip}/checkpoint_{epoch}_NP.pt")
        info_dict['fopt'] = training_loss
        return info_dict
        
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
        