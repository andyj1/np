#!/usr/bin/env python3

import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn import gaussian_process as gp
from tqdm import tqdm

import anp_utils
import model
from bayes_opt_custom import util
import data


class SurrogateModel(object):
    def __init__(self, cfg, device):
        super(SurrogateModel, self).__init__()
        self.device = device
        self.cfg = cfg # train config
        self.model_type = cfg['model_type']
        self.random_state = cfg['random_state']

        self.context_size = self.cfg['context_size']
        self.num_samples = self.cfg['num_samples']
        self.n_epoch = self.cfg['n_epoch']
        self.n_epoch_subsequent = self.cfg['n_epoch_subsequent']
        self.test_interval = 1 # set differently by n_epoch in training
        self.lr = self.cfg['lr']
        
        self.initial_training = True
        
        self.fig = plt.figure()
        
        self._random_state = util.ensure_rng(self.random_state)
        # initialize model
        if 'gp' not in self.model_type:
            # neural processes family model
            if self.model_type == 'anp':
                self.model = model.AttentiveNP(self.cfg).to(self.device)
            elif self.model_type == 'mlp':
                self.model = model.FeedforwardMLP(self.cfg).to(self.device)
            # need optimizer for neural processes
            self.optimizer_cls = optim.Adam
            self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)
            
        elif self.model_type == 'gp':
            # scikit learn GPR
            self.model = gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=2.5), 
                                                  alpha=1e-6, 
                                                  normalize_y=True, 
                                                  n_restarts_optimizer=5, 
                                                  random_state=self._random_state)
    # preprocess list of values into dataloaders 
    # to be used in BO::suggest()
    def preprocess(self, xlist, ylist):
        x = pd.DataFrame(xlist, columns=['x1', 'x2']).astype(np.float32)
        y = pd.DataFrame(ylist, columns=['y']).astype(np.float32)
        df =  pd.concat((x, y), axis=1)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1:].values
        dataset = data.TemplateDataset(x, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=self.cfg['num_workers'])
        test_loader = None
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader=None):
        if self.cfg['visualize']:
            save_img_path = f'./fig/{self.cfg["dataset"]}/{self.model_type}'
            os.makedirs(save_img_path, exist_ok=True, mode=0o755)
        
        if self.initial_training:
            self.initial_training = False
            init_epoch = 1
            n_epoch = self.n_epoch
        else:
            init_epoch = self.n_epoch + 1
            n_epoch = self.n_epoch + self.n_epoch_subsequent
        self.model.train()
        self.test_interval = max(1, (n_epoch+1-init_epoch) // 5)
        
        t = tqdm(range(init_epoch, n_epoch+1, 1), total=(n_epoch+1-init_epoch))
        for epoch in t:
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                
                num_samples, input_dim = x.shape # batch size = 1
                context_idx, target_idx = anp_utils.context_target_split(num_samples, self.context_size)
                x = x.expand(1, -1, -1)
                y = y.expand(1, -1, -1) # [B, Num_samples, num_dim]
                x_context, y_context = x[:, context_idx, :], y[:,  context_idx, :]
                x_target, y_target = x[:, target_idx, :], y[:, target_idx, :]
                
                repeat_bs = 16 # repeat along batch dim
                x_context = x_context.repeat(repeat_bs, 1, 1).expand(repeat_bs, -1, -1)
                y_context = y_context.repeat(repeat_bs, 1, 1).expand(repeat_bs, -1, -1)
                x_target = x_target.repeat(repeat_bs, 1, 1).expand(repeat_bs, -1, -1)
                y_target = y_target.repeat(repeat_bs, 1, 1).expand(repeat_bs, -1, -1)
                x_all = torch.cat([x_context, x_target], dim=1)
                y_all = torch.cat([y_context, y_target], dim=1)
                # print('[input to model]:\t',x_context.shape, y_context.shape, x_all.shape, y_all.shape)
                self.optimizer.zero_grad()
                
                query = (x_context, y_context, x_all)
                mu, sigma, log_p, kl, loss = self.model(query, y_all)
                
                loss.backward()
                self.optimizer.step()
            
            if self.cfg['verbose']:
                if self.model_type == 'anp':
                    print('Epoch: {:<5} | loss: {:.6f}\tKLD: {:.6f}\tmu: {:.4f},sigma: {:.4f}'.format(epoch, loss.item() / repeat_bs, kl.mean(), mu.mean(), sigma.mean()))
                else:
                    print('Epoch:{:<5}\tloss: {:.6f}'.format(epoch, loss.item() / repeat_bs))
                    
            if self.cfg['visualize']:                
                # visusalize negative because we deal with minimizing problem (and the framework is set to maximize the target)                
                if epoch == init_epoch or epoch % self.test_interval == 0:
                    plt.ion()
                    plt.clf()
                    if input_dim == 1:
                        anp_utils.plot_functions(x_all.cpu().detach(),
                                                -y_all.cpu().detach(),
                                                x_context.cpu().detach(),
                                                -y_context.cpu().detach(),
                                                -mu.cpu().detach(),
                                                sigma.cpu().detach())
                    elif input_dim == 2:
                        ax = self.fig.add_subplot(111, projection='3d')
                        ax = anp_utils.plot_functions_2d(x_all.cpu().detach(),
                                                        -y_all.cpu().detach(),
                                                        x_context.cpu().detach(),
                                                        -y_context.cpu().detach(),
                                                        -mu.cpu().detach(),
                                                        sigma.cpu().detach(),
                                                        ax)
                    ax.view_init(elev=15, azim=-45)
                    plt.draw()
                    title_str = f'{self.model_type.upper()}, epoch ' + str(epoch)
                    plt.title(title_str)

                    if epoch % self.test_interval == 0:
                        plt.savefig(f'./fig/{self.cfg["dataset"]}/{self.model_type}/anp_{n_epoch}.png', transparent=False, edgecolor='none')
                    plt.pause(0.0000001)
        self.n_epoch = n_epoch
    