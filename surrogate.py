#!/usr/bin/env python3

import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import SGD
from tqdm import tqdm, trange

from custom.fit import fit_gpytorch_torch
from NPModels.ANP import ANP as AttentiveNeuralProcesses
from NPModels.NP import NP as NeuralProcesses
from utils.np_utils import random_split_context_target
from utils.utils import checkParamIsSentToCuda

# color codes for text in UNIX shell
CRED = '\033[91m'
CGREEN = '\033[92m'
CCYAN = '\033[93m'
CBLUE = '\033[94m'
CEND = '\033[0m'

def save_ckpt(model, optimizer, np_bool, chip, iter):
    checkpoint = {'state_dict': model.state_dict()}
    if optimizer is not None:
        checkpoint.update({'optimizer' : optimizer.state_dict()})
    base_path = f'ckpts/'
    chip_path = f'{chip}/' if chip is not 'toy' else 'toy/'
    model_type = 'NP' if np_bool else 'GP'
    checkpoint_path = f'_iter_{iter}'
    extension = '.pt'
    
    save_path = base_path + chip_path + model_type + checkpoint_path + extension
    torch.save(checkpoint, save_path)

class SurrogateModel(object):
    def __init__(self, train_X, train_Y, args, cfg, writer, epochs, model_type):
        super(SurrogateModel, self).__init__()
        self.device = args.device
        self.writer = writer        
        self.epochs = epochs
        self.model_type = model_type
        self.DISPLAY_FOR_EVERY = cfg['train']['display_for_every']
        if self.model_type.lower() == 'np':
            self.num_context = cfg[self.model_type.lower()]['num_context']
        
        # configure optimizer for training GP (fit_gpytorch_torch); doesn't support dense gradients like SparseAdam
        if cfg['train']['optimizer'] == 'Adam':
            self.optimizer_cls = optim.Adam
        elif cfg['train']['optimizer'] == 'SGD':
            self.optimizer_cls = optim.SGD
        self.lr = cfg['train']['lr']

        # initialize model
        self.model, mll, self.optimizer = self.initialize_model(cfg, self.model_type, train_X, train_Y, self.device)
    
    def initialize_model(self, cfg, model_type, train_X, train_Y, device):
        model = None
        mll = None
        if model_type == 'NP':
            model = NeuralProcesses(cfg, device)
            model = model.to(device)
        elif model_type == 'ANP':
            model = AttentiveNeuralProcesses(cfg)
            model = model.to(device)
        elif self.model_type == 'GP':
            model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            # model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
            mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            mll.to(train_X)
        
        optimizer = self.optimizer_cls(model.parameters(), lr=self.lr)
        return model, mll, optimizer
    
    def train(self, inputs, targets, args, cfg, toy_bool, iter=0):    
        if 'GP' in args.model.upper():
            info_dict = self.fitGP(inputs[:, 0:2], targets, cfg, toy_bool=toy_bool, iter=iter)
        elif 'NP' in args.model.upper():
            info_dict = self.fitNP(inputs[:, 0:2], targets, cfg, toy_bool=toy_bool)
    
    # ======================================================================================================================
    # custom GP fitting
    def fitGP(self, train_X, train_Y, cfg, toy_bool=False, iter=0):
        chip = cfg['MOM4']['parttype']
        
        self.model, mll, self.optimizer = self.initialize_model(cfg, self.model_type, train_X, train_Y, self.device)
        mll.train()
        
        # customize optimizer in 'fit.py' in fit_gpytorch_torch()
        # optimizer need not have a closure
        optimizer_options = {'lr': cfg['train']['lr'], 'maxiter': cfg['train']['maxiter'], 'disp': cfg['train']['disp']}
        
        ''' define custom optimizer using optimizer class: "self.optimizer_cls" '''
        # self.optimizer = self.optimizer_cls(model.parameters())
        self.optimizer = None # if None, defines a new optimizer within fit_gpytorch_torch
        # mll, info_dict, self.optimizer = fit_gpytorch_torch(mll=mll, \
        #                                     optimizer_cls=self.optimizer_cls, \
        #                                     options=optimizer_options, \
        #                                     approx_mll=True, \
        #                                     custom_optimizer=self.optimizer, \
        #                                     display_for_every=self.DISPLAY_FOR_EVERY)
        # loss = info_dict['fopt']
        
        # alternative to fit_gpytorch_torch; more general fit API
        # fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch)
        mll = fit_gpytorch_model(mll)
        info_dict = {}
        
        

        
        # uncomment the following for custom GP fitting \
        # * can't output lengthscale value due to >1 inputs
        
        # optimizer_options = {'lr': 1e-3} # for pytorch
        # self.model.train() # set train mode
        # self.optimizer = self.optimizer_cls([{'params': self.model.parameters()}], **optimizer_options)        
        # DISPLAY_FOR_EVERY = self.epochs
        # t = trange(self.epochs, desc='', leave=False)
        # for train_epoch in t:

        #     self.optimizer.zero_grad()
        #     output = self.model(train_X)
        #     loss = -mll(output, self.model.train_targets)
        #     loss.backward()
        #     t.set_description(f"[Train] Iter {train_epoch+1:>3}/{self.epochs} - Loss: {loss.item():>4.3f} - noise: {self.model.likelihood.noise.item():>4.3f}\n",refresh=False)
            
        #     # if epoch % DISPLAY_FOR_EVERY == 0:
        #     #     print(f"[Train] Iter {epoch+1:>3}/{self.epochs} - Loss: {loss.item():>4.3f} - noise: {model.likelihood.noise.item():>4.3f}")
        #     # print(f"lengthscale: {model.covar_module.base_kernel.lengthscale:>4.3f}")
        #     self.optimizer.step()
        
        # syntax: save_ckpt(model, optimizer, toy_bool, np_bool, chip, iter)
        save_ckpt(self.model, self.optimizer, False, chip, iter)
        self.model = mll.model
        return info_dict
    
    def fitNP(self, train_X, train_Y, cfg, toy_bool=False, iter=0):
        chip = cfg['MOM4']['parttype']
        n_iter = cfg['np']['num_iter']
        self.DISPLAY_FOR_EVERY = cfg['train']['display_for_every']
        info_dict = {}
        
        # re-initialize a new model to train
        self.model, _, self.optimizer = self.initialize_model(cfg, self.model_type, train_X, train_Y, self.device)
        self.model.to(self.device)
        
        t = tqdm(range(self.epochs), total=self.epochs)
        for train_epoch in t:
            epoch_losses = []
                
            num_samples = train_X.shape[0]
            assert num_samples > self.num_context
            
            # split data into context and target
            context_x, context_y, target_x, target_y = random_split_context_target(train_X, train_Y, self.num_context)
            # print(f'[INFO] context_x: {context_x.shape}, context_y:{context_y.shape}, target_x:{target_x.shape}, target_y:{target_y.shape}')
            
            # upload to gpu
            context_x = context_x.to(self.device)
            context_y = context_y.to(self.device)
            target_x = target_x.to(self.device)
            target_y = target_y.to(self.device)
                    
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            target_x = target_x.unsqueeze(0)
            target_y = target_y.unsqueeze(0)
            query = (context_x, context_y, target_x)
            
            # training
            # ** need to iterate for performance
            t_iter = tqdm(range(n_iter), total=n_iter)
            for iter in t_iter:
                # set model and optimizer
                self.model.train()
                # self.adjust_learning_rate(init_lr=self.lr, optimizer=self.optimizer, step_num=train_epoch+1)
                
                context_mu, logits, p_y_pred, q_target, q_context, loss, kl = self.model(query, target_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                
                # update progress bar for training
                # t_iter.set_description(f'iter: {iter+1}/{n_iter} / train loss: {CRED}{loss.item():10.5f}{CEND} / kl: {CBLUE}{kl.item():10.5f}{CEND}\n')
                
                ''' evaluate '''
                # if True:
                    # if iter % self.DISPLAY_FOR_EVERY == 0:
                
                # if len(epoch_losses) > 2 or len(epoch_losses) == 0:
                #     if iter % self.DISPLAY_FOR_EVERY == 0 and (epoch_losses[-1] <= epoch_losses[-2]) and (epoch_losses[-1] <= epoch_losses[-3]) \
                #         and (epoch_losses[-1] <= epoch_losses[-4]) and (epoch_losses[-1] <= epoch_losses[-5]):
                            
                #         self.model.eval()
                #         with torch.no_grad():
                #             p_y_pred, pred_y, pred_std = self.model(query)
                #             print('\n\n')
                #             print('current training kl: %3.5f' % (kl))
                #             print(f'epoch: {train_epoch}, iter: {iter} | SAMPLE 0: target_x: ({target_x[0, 0, 0].item():3.5f},{target_x[0, 0, 1].item():3.5f}), target_y: {CBLUE}{target_y[0,0,0].item():3.5f}{CEND}, pred_y: {CRED}{pred_y[0, 0, 0].item():3.5f}{CEND}, pred_sigma: {pred_std[0, 0, 0].item():3.5f}')
                #             print(f'\t\t     | SAMPLE 1: target_x: ({target_x[0, 1, 0].item():3.5f},{target_x[0, 1, 1].item():3.5f}), target_y: {CBLUE}{target_y[0,1,0].item():3.5f}{CEND}, pred_y: {CRED}{pred_y[0, 1, 0].item():3.5f}{CEND}, pred_sigma: {pred_std[0, 1, 0].item():3.5f}')
                #             # time.sleep(1)
                # if (iter > 1000) and (epoch_losses[-1] <= epoch_losses[-2]) and (epoch_losses[-1] <= epoch_losses[-3]) \
                #         and (epoch_losses[-1] <= epoch_losses[-4]) and (epoch_losses[-1] <= epoch_losses[-5]):
                #     break

            self.writer.add_scalar(f"Loss/train_NP_{cfg['train']['num_samples']}_samples_fitNP", loss.item(), train_epoch)
        
        # save_ckpt(self.model, self.optimizer, True, chip, iter)

        info_dict['fopt'] = epoch_losses[-1]
        return info_dict
    
        
    def adjust_learning_rate(self, init_lr, optimizer, step_num, warmup_step=4000):
        lr = init_lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    