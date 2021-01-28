#!/usr/bin/env python3

import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import SGD
from tqdm import trange

# from botorch.fit import fit_gpytorch_model
from custom.fit import fit_gpytorch_torch
from NPModels.ANP import ANP as AttentiveNeuralProcesses
from NPModels.NP import NP as NeuralProcesses
from utils.np_utils import random_split_context_target


def save_ckpt(model, optimizer, toy_bool, np_bool, chip, iter):
    checkpoint = {'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()}
    base_path = f'ckpts/'
    chip_path = f'{chip}/' if chip is not 'toy' else 'toy/'
    
    checkpoint_path = f'checkpoint_{iter}_'
    model_type = 'NP' if np_bool else 'GP'
    extension = '.pt'
    
    save_path = base_path + chip_path + checkpoint_path + model_type + extension
    torch.save(checkpoint, save_path)
        
class SurrogateModel(object):
    def __init__(self, train_X, train_Y, args, cfg, writer, device=torch.device('cpu'), epochs=100, model_type='GP'):
        super(SurrogateModel, self).__init__()
        
        start_time = time.time()
        
        self.epochs = epochs
        self.model_type = model_type
        if self.model_type == 'NP':
            self.model = NeuralProcesses(cfg).to(device)
        elif self.model_type == 'ANP':
            self.model = AttentiveNeuralProcesses(cfg).to(device)
        elif self.model_type == 'GP':
            self.model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            self.model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
            mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
            mll.to(train_X)
        
        self.DISPLAY_FOR_EVERY = cfg['train']['display_for_every']
        # optimizer_cls = optim.AdamW
        # optimizer_cls = optim.SparseAdam # doesn't support dense gradients
        # self.optimizer_cls = optim.Adamax
        self.optimizer_cls = optim.Adam
        self.optimizer = self.optimizer_cls(self.model.parameters())
        
        # use GPU if available
        self.device = device
        self.dtype = torch.float
        self.writer = writer

        end_time = time.time()
        print(': took %.3f seconds' % (end_time-start_time))
        
    # custom GP fitting
    def fitGP(self, train_X, train_Y, cfg, toy_bool=False, iter=0):
        chip = cfg['MOM4']['parttype']
        
        # re-initialize model because GPyTorch's SingleTaskGP taks in X, Y at the initialization
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
                                            device=self.device,
                                            display_for_every=self.DISPLAY_FOR_EVERY)
        loss = info_dict['fopt']
        self.model = mll.model
        
        # alternative to fit_gpytorch_torch; more general fit API
        # fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch)

        ''' 
        # uncomment the following for custom GP fitting \
        # * can't output lengthscale value due to >1 inputs
        
        optimizer_options = {'lr': 0.05} # for pytorch
        model.train() # set train mode
        self.optimizer = self.optimizer_cls([{'params': model.parameters()}], **optimizer_options)        
        DISPLAY_FOR_EVERY = self.epochs
        t = trange(self.epochs, desc='', leave=False)
        for train_epoch in t:

            self.optimizer.zero_grad()
            output = model(train_X)
            loss = -mll(output, model.train_targets)
            loss.backward()
            # t.set_description(f"[Train] Iter {train_epoch+1:>3}/{self.epochs} - Loss: {loss.item():>4.3f} - noise: {model.likelihood.noise.item():>4.3f}\n",refresh=False)
            
            # if epoch % DISPLAY_FOR_EVERY == 0:
            #     print(f"[Train] Iter {epoch+1:>3}/{self.epochs} - Loss: {loss.item():>4.3f} - noise: {model.likelihood.noise.item():>4.3f}")
            # print(f"lengthscale: {model.covar_module.base_kernel.lengthscale:>4.3f}")
            self.optimizer.step()
        '''
        # syntax: save_ckpt(model, optimizer, toy_bool, np_bool, chip, iter)
        save_ckpt(self.model, self.optimizer, toy_bool, False, chip, iter)
        return info_dict
    
    def fitNP(self, train_X, train_Y, cfg, toy_bool=False, iter=0):
        chip = cfg['MOM4']['parttype']
        info_dict = {}
        # not re-initializing the model (already defined at self.model)
        self.model.train()
        lr = cfg['train']['lr']
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        num_context = cfg[self.model_type.lower()]['num_context']
        
                
        for train_epoch in range(self.epochs):
            loss = 0.0
            
            self.adjust_learning_rate(optimizer, train_epoch+1)
            
            optimizer.zero_grad()
            # split into context and target
            x_context, y_context, x_target, y_target = random_split_context_target(train_X, train_Y, num_context)

            # print(f'[INFO] x_context: {x_context.shape}, y_context:{y_context.shape}, x_target:{x_target.shape}, y_target:{y_target.shape}')
            
            # send to gpu
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)

            # forward pass
            mu, sigma, log_p, kl, loss_val = self.model(x_context, y_context, x_target, y_target)
            # self.writer.add_scalar(f"Loss/train_NP_{cfg['train']['num_samples']}_samples_fitNP", loss.item(), train_epoch)
            
            loss += -loss_val
            loss = loss.mean()
            # backprop
            # training_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            if (train_epoch+1) % (self.epochs//self.DISPLAY_FOR_EVERY) == 0:
                print('[INFO] train_epoch: {}/{}, \033[94m loss: {} \033[0m'.format(train_epoch+1, self.epochs, loss.item()))
        
        
        save_ckpt(self.model, self.optimizer, toy_bool, True, chip, iter)

        info_dict['fopt'] = loss.mean().item()
        return info_dict

    # '''
    # evaluate: performs evaluation at test points and return mean and lower, upper bounds
    # '''
    # def evaluate(self, test_X):
    #     posterior, variance = None, None
    #     if self.model is not None:
    #         self.model.eval() # set eval mode
    #         with torch.no_grad():
    #             posterior = self.model.posterior(test_X)
    #             # upper and lower confidence bounds (2 standard deviations from the mean)
    #             lower, upper = posterior.mvn.confidence_region()
    #         return posterior.mean.cpu().numpy(), (lower.cpu().numpy(), upper.cpu().numpy())

    
    def adjust_learning_rate(self, optimizer, step_num, warmup_step=4000):
        lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr