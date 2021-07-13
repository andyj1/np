import argparse
import gc
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import anp_utils
from stringcase import pascalcase
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import tqdm

class NPTrainer():
    def __init__(self, model, context_size, optimizer, dataset, device):
        self.model = model
        self.context_size = context_size
        self.device = device
        self.optimizer = optimizer
        self.epoch_losses = []
        self.dataset = dataset # dataset name
        os.makedirs(f'./fig/{self.dataset}', exist_ok=True, mode=0o755)
        print('running on:', self.device)
        
    def train(self, train_loader, test_loader, n_epoch, test_interval):
        self.model = self.model.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        
        visualize_bool = True # visualize initial set of data before training
        visualize_bool = False
        
        fig = plt.figure()
        ''' train '''
        self.model.train()
        for epoch in range(1, n_epoch+1, 1):
            plt.ion()
            plt.clf()
            train_loss_sum = 0.
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                # print(x.shape, y.shape)
                
                num_samples, input_dim = x.shape # batch size = 1
                context_idx, target_idx = utils.context_target_split(num_samples, self.context_size) # split along dim=1
                x = x.expand(1, -1, -1)
                y = y.expand(1, -1, -1) # [B, Num_samples, num_dim]
                x_context, y_context = x[:, context_idx, :], y[:,  context_idx, :]
                x_target, y_target = x[:, target_idx, :], y[:, target_idx, :]
                # print('[split]:\t', x_context.shape, y_context.shape, x_target.shape, y_target.shape)
                
                # ==== visualize context and target(=all) ====
                if visualize_bool:
                    x_all = torch.cat([x_context, x_target], dim=1) # dim 1: num_samples
                    y_all = torch.cat([y_context, y_target], dim=1) # dim 1: num_samples
                    
                    if input_dim == 1:    
                        order = torch.argsort(x, dim=1)
                        x = x.squeeze()[order].squeeze()
                        y = y.squeeze()[order].squeeze()

                        order = torch.argsort(x_context, dim=1)
                        x_context = x_context.squeeze()[order].squeeze()
                        y_context = y_context.squeeze()[order].squeeze()

                        order = torch.argsort(x_all, dim=1)
                        x_all = x_all.squeeze()[order].squeeze()
                        y_all = y_all.squeeze()[order].squeeze()
                        # print('[target(all)]:', x_all.shape, y_all.shape)

                        # plot 1d
                        plt.plot(x.cpu(), y.cpu(), 'k--', linewidth=0.5, label='all')
                        # plt.plot(x_target, y_target, 'b.', markersize=1, label='target')
                        plt.plot(x_all.cpu(), y_all.cpu(), 'b.--', markersize=3, linewidth=1, label='target')
                        plt.plot(x_context.cpu(), y_context.cpu(), 'go', markersize=6, linewidth=1, label='context')
                        plt.legend()
                        plt.title('ANP, epoch='+str(epoch))               
                        # plt.pause(0.1)
                        
                    if input_dim == 2:
                        x_all = x_all[0].unsqueeze(0)
                        y_all = y_all[0].unsqueeze(0)
                        context_x = x_context[0].unsqueeze(0)
                        context_y = y_context[0].unsqueeze(0)
                        
                        order = torch.argsort(x_all, dim=1)[0,:,0]
                        x_all = x_all[:, order, :]
                        y_all = y_all[:, order, :]
                        
                        order = torch.argsort(context_x, dim=1)[0, :, 0]
                        context_x = context_x[:, order, :]
                        context_y = context_y[:, order, :]                        
                        
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(x_all[0, :, 0].cpu(), x_all[0, :, 1].cpu(), y_all[0, :, 0].cpu(), c='b', label='target')
                        ax.scatter(context_x[0, :, 0].cpu(), context_x[0, :, 1].cpu(), context_y[0, :, 0].cpu(), c='g', label='context')
                        
                        ax.set_xlabel('x1')
                        ax.set_ylabel('x2')
                        ax.set_zlabel('y')
                        ax.legend()
                        ax.grid('off')
                        ax.view_init(elev=30, azim=-60)
                    plt.show()
                    
                    plt.pause(5)
                    plt.clf()
                    # sys.exit(1)
                # ==== end visualize ====
                
                repeat_bs = 16 # repeat along batch dim
                x_context = x_context.repeat(repeat_bs, 1, 1).expand(repeat_bs, -1, -1)
                y_context = y_context.repeat(repeat_bs, 1, 1).expand(repeat_bs, -1, -1)
                x_target = x_target.repeat(repeat_bs, 1, 1).expand(repeat_bs, -1, -1)
                y_target = y_target.repeat(repeat_bs, 1, 1).expand(repeat_bs, -1, -1)
                x_all = torch.cat([x_context, x_target], dim=1) # dim 1: num_samples
                y_all = torch.cat([y_context, y_target], dim=1) # dim 1: num_samples
                # print('[input to model]:\t',x_context.shape, y_context.shape, x_all.shape, y_all.shape)
                
                # print('[input to model]:\t',x_context.shape, y_context.shape, x_all.shape, y_all.shape)
                # print(x_context[0:5], x_target[0:5])
                # sys.exit(1)
            
                self.optimizer.zero_grad()
                
                query = (x_context, y_context, x_all)
                mu, sigma, log_p, kl, loss = self.model(query, y_all)
                
                train_loss_sum += loss.item()
                # print('[forward passed]', mu.shape, sigma.shape, loss.item())
                loss.backward()
                self.optimizer.step()
                
                # neglect the following print statement (bc batch size is 1)
                # print('Epoch: {} [batch {}/{}]\tloss: {:.6f}, KLD: {:.6f}'.format( \
                #         epoch+1, batch_idx * len(y_all), len(train_loader.dataset), \
                        # loss.item() / len(y_all), kl.mean()))
            # if epoch % test_interval == 0:
                # print(loss.item())
            # repeat_bs = len(y_all)
            print('Epoch:{:<5}\tloss: {:.6f}, KLD: {:.6f}'.format(epoch, loss.item() / repeat_bs, kl.mean()))
                
            # train_loss_sum /= len(train_loader.dataset)
            # print('\t[Train] avg loss: {:.4f}'.format(train_loss_sum))
            
            # print(mu.shape, sigma.shape)
            if epoch == 1 or epoch % test_interval == 0:                
                print('..........................................')
                if input_dim == 1:
                    utils.plot_functions(x_all.cpu().detach(),
                                        y_all.cpu().detach(),
                                        x_context.cpu().detach(),
                                        y_context.cpu().detach(),
                                        mu.cpu().detach(),
                                        sigma.cpu().detach())
                elif input_dim == 2:
                    ax = fig.add_subplot(111, projection='3d')
                    ax.view_init(elev=30, azim=-60)
                    ax = utils.plot_functions_2d(x_all.cpu().detach(),
                                            y_all.cpu().detach(),
                                            x_context.cpu().detach(),
                                            y_context.cpu().detach(),
                                            mu.cpu().detach(),
                                            sigma.cpu().detach(),
                                            ax)
                    ax.view_init(elev=30, azim=-60)
                title_str = 'ANP, epoch ' + str(epoch)
                plt.title(title_str)

                # if epoch == 1 or epoch % 50 == 0:
                #     plt.savefig(f'./fig/{self.dataset}/anp_epoch_{epoch}.png')
                plt.pause(0.01)
                
        if visualize_bool:
            plt.ioff()
            plt.show() 