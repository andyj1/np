#!/usr/bin/env python3

import random
import sys
import time
from utils.utils import checkParamIsSentToCuda

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import SGD
from tqdm import tqdm, trange

# from botorch.fit import fit_gpytorch_model
from custom.fit import fit_gpytorch_torch
from NPModels.ANP import ANP as AttentiveNeuralProcesses
from NPModels.NP import NP as NeuralProcesses
from utils.np_utils import random_split_context_target


def save_ckpt(model, optimizer, np_bool, chip, iter):
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
                
        # use GPU if available
        self.device = device
        self.dtype = torch.float
        self.writer = writer
        
        self.epochs = epochs
        self.model_type = model_type
        self.model, mll = self.initialize_model(cfg, self.model_type, train_X, train_Y, self.device)
        self.DISPLAY_FOR_EVERY = cfg['train']['display_for_every']
        self.num_context = cfg[self.model_type.lower()]['num_context']
        
        # configure optimizer
        if cfg['train']['optimizer'] == 'Adam':
            self.optimizer_cls = optim.Adam
        elif cfg['train']['optimizer'] == 'SGD':
            self.optimizer_cls = optim.SGD
        # optimizer_cls = optim.AdamW
        # optimizer_cls = optim.SparseAdam # doesn't support dense gradients
        # self.optimizer_cls = optim.Adamax

        # initialize optimizer
        self.lr = cfg['train']['lr']
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)
        
    # custom GP fitting
    def fitGP(self, train_X, train_Y, cfg, toy_bool=False, iter=0):
        chip = cfg['MOM4']['parttype']
        
        self.model, mll = self.initialize_model(cfg, self.model_type, train_X, train_Y, self.device)
        # default wrapper tor training
        # fit_gpytorch_model(mll)
        mll.train()
        
        # customize optimizer in 'fit.py' in fit_gpytorch_torch()
        # optimizer need not have a closure
        optimizer_options = {'lr': cfg['train']['lr'], 'maxiter': cfg['train']['num_iter'], 'disp': True}
        
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
        save_ckpt(self.model, self.optimizer, False, chip, iter)
        return info_dict
    
    def fitNP(self, train_X, train_Y, cfg, toy_bool=False, iter=0):
        chip = cfg['MOM4']['parttype']
        n_iter = cfg['np']['n_iter']
        test_for_every = cfg['train']['display_for_every']
        info_dict = {}
        
        # re-initialize a new model to train
        self.model, _ = self.initialize_model(cfg, self.model_type, train_X, train_Y, self.device)
        self.model.to(self.device)
        
        t = tqdm(range(self.epochs), total=self.epochs)
        for train_epoch in t:
            epoch_loss = 0.
            
            # split data into context and target
            context_x, context_y, target_x, target_y = random_split_context_target(train_X, train_Y, self.num_context)
            # print(f'[INFO] context_x: {context_x.shape}, context_y:{context_y.shape}, target_x:{target_x.shape}, target_y:{target_y.shape}')
            
            # upload to gpu
            context_x = context_x.to(self.device)
            context_y = context_y.to(self.device)
            target_x = target_x.to(self.device)
            target_y = target_y.to(self.device)
            query = (context_x, context_y, target_x)
            
            # training
            # ** need to iterate for performance
            t_iter = tqdm(range(n_iter), total=n_iter)
            for iter in t_iter:
                # set model and optimizer
                self.model.train()
                self.adjust_learning_rate(init_lr=self.lr, optimizer=self.optimizer, step_num=train_epoch+1)
                self.optimizer.zero_grad()
                
                context_mu, logits, p_y_pred, q_target, q_context, loss = self.model(query, target_y)
                
                try:
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                except ValueError as ve:
                    print('[ERROR] loss is None...')
                
                t_iter.set_description('iter: %d/%d / loss: %.f' % (iter, n_iter,epoch_loss/int(iter+1)))
                # print('  loss:',loss.item())
            
            # self.writer.add_scalar(f"Loss/train_NP_{cfg['train']['num_samples']}_samples_fitNP", loss.item(), train_epoch)
        
        # save_ckpt(self.model, self.optimizer, True, chip, iter)

        # info_dict['fopt'] = epoch_loss.item()
        return info_dict
    
        
    def adjust_learning_rate(self, init_lr, optimizer, step_num, warmup_step=4000):
        lr = init_lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def initialize_model(self, cfg, model_type, train_X, train_Y, device):
        model = None
        mll = None
        if model_type == 'NP':
            model = NeuralProcesses(cfg, device)
        # elif model_type == 'ANP':
        #     model = AttentiveNeuralProcesses(cfg)
        elif self.model_type == 'GP':
            model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
            mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            mll.to(train_X)
        
        return model, mll