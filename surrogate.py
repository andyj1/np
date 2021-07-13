#!/usr/bin/env python3

import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
import os

import model, anp_utils


class SurrogateModel(object):
    def __init__(self, cfg, model_type, device):
        super(SurrogateModel, self).__init__()
        self.device = device
        self.cfg = cfg
        self.model_type = model_type

        self.context_size = self.cfg['context_size']
        self.num_samples = self.cfg['num_samples']
        self.n_epoch = self.cfg['n_epoch']
        self.test_interval = self.cfg['test_interval']
        self.lr = self.cfg['lr']
        
        save_img_path = f'./fig/{self.cfg["dataset"]}_{self.model_type}'
        os.makedirs(save_img_path, exist_ok=True, mode=0o755)
    
    def train(self, train_loader, test_loader):
        if self.model_type == 'anp':
            self.model = model.AttentiveNP(self.cfg).to(self.device)
        elif self.model_type == 'mlp':
            self.model = model.FeedforwardMLP(self.cfg).to(self.device)

        # if self.cfg['train']['optimizer'] == 'Adam':
        self.optimizer_cls = optim.Adam
        self.optimizer = self.optimizer_cls(self.model.parameters(), 
                                            lr=self.lr)
        
        self.fit(train_loader, test_loader)
        
    
    def fit(self, train_loader, test_loader):
        self.model.train()
        fig = plt.figure()
        t = tqdm(range(1, self.n_epoch+1, 1), total=self.n_epoch)
        for epoch in t:
            plt.ion()
            plt.clf()
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                # print(x.shape, y.shape)
                
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
                
            if self.model_type == 'anp':
                print('Epoch: {:<5} | loss: {:.6f}\tKLD: {:.6f}\tmu: {:.4f},sigma: {:.4f}'.format(epoch, loss.item() / repeat_bs, kl.mean(), mu.mean(), sigma.mean()))
            else:
                print('Epoch:{:<5}\tloss: {:.6f}'.format(epoch, loss.item() / repeat_bs))
            if epoch == 1 or epoch % self.test_interval == 0:
                if input_dim == 1:
                    anp_utils.plot_functions(x_all.cpu().detach(),
                                            y_all.cpu().detach(),
                                            x_context.cpu().detach(),
                                            y_context.cpu().detach(),
                                            mu.cpu().detach(),
                                            sigma.cpu().detach())
                elif input_dim == 2:
                    ax = fig.add_subplot(111, projection='3d')
                    ax = anp_utils.plot_functions_2d(x_all.cpu().detach(),
                                                    y_all.cpu().detach(),
                                                    x_context.cpu().detach(),
                                                    y_context.cpu().detach(),
                                                    mu.cpu().detach(),
                                                    sigma.cpu().detach(),
                                                    ax)
                ax.view_init(elev=30, azim=-45)
                plt.draw()
                title_str = f'{self.model_type.upper()}, epoch ' + str(epoch)
                plt.title(title_str)

                # if epoch % 100 == 0:
                #     plt.savefig(f'./fig/{self.cfg["dataset"]}_{self.model_type}/anp_{epoch}.png', transparent=False, edgecolor='none')
                # plt.pause(0.001)
        
    def adjust_learning_rate(self, init_lr, optimizer, step_num, warmup_step=4000):
        lr = init_lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    