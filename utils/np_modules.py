#!/usr/bin/env python3

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicEncoder(nn.Module):
    """
    (x,y) --> aggregated representation r
    """
    def __init__(self, x_dim, y_dim, r_dim):
        super(DeterministicEncoder, self).__init__()
 
        self.fc1 = nn.Linear(x_dim + y_dim, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, r_dim)

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=0)
        
    def forward(self, x, y):
        
        x = x.squeeze(0)
        y = y.squeeze(0)
        r = torch.cat([x, y], dim=1)
        
        r = F.relu(self.fc1(r))
        r = F.relu(self.fc2(r))
        r = self.fc3(r)

        # aggregate r
        r_aggregated = self.aggregate(r)
        
        return r_aggregated

class LatentEncoder(nn.Module):
    """
    aggregated representation --> latent variable z
    """
    def __init__(self, r_dim, z_dim):
        super(LatentEncoder, self).__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.fc_mu = nn.Linear(r_dim, z_dim)
        self.fc_sigma = nn.Linear(r_dim, z_dim)
    
    def forward(self, r):
        # print('r shape:',r.shape)
        # print('r_dim:', self.r_dim, 'z_dim:', self.z_dim)
        # print('r->z dim fc mu shape:',self.fc_mu(r).shape)
        
        # print('fc sigma shape:',self.fc_sigma(r).shape)
        # print('softplus shape:',F.softplus(self.fc_sigma(r)).shape)
        mu, sigma = self.fc_mu(r), F.softplus(self.fc_sigma(r))
        # print('[Latent Encoder] mu:', mu.shape, 'sigma:', sigma.shape)
        return mu, sigma
        
class Decoder(nn.Module):
    """
    target x and latent variable z --> posterior prediction p(f(x) | x,z)
    """
    def __init__(self, x_dim, y_dim, z_dim):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(x_dim + z_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        
        self.fc_mu = nn.Linear(32, y_dim)
        self.fc_sigma = nn.Linear(32, y_dim)
    
    def forward(self, x, z):
        # Concatenate to a tensor of shape [len(x), len(z), x_dim + z_dim]
        '''
        [1] x shape: torch.Size([784, 2])
        [2] x shape: torch.Size([784, 10, 2])
        [3] z shape: torch.Size([10, 5])
        [4] z shape: torch.Size([784, 10, 5])
        [5] y shape: torch.Size([784, 10, 7])
        '''
        # num_target (24*24=784)
        # number of z_samples (elbo_samples = z_dim*2)
        
        # print('[1] orig x shape:', x.shape) # [batch_size, num_target, x_dim]
        
        x = x.expand(z.shape[0], -1, -1).transpose(0, 1)
        # print('[2] x shape:', x.shape) # [num_target, z_dim*2, x_dim]
        # print('[3] orig z shape:',z.shape) # [z_dim*2, num_target]
        
        z = z.unsqueeze(0).expand(x.shape[0], -1, -1)
        # print('[4] z shape:',z.shape) # [num_target, z_dim*2, num_target]
        
        y = torch.cat([x, z], dim=-1)
        # print('[4] y shape:', y.shape) # [num_target, z_dim*2, x_dim+num_target]
        
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        
        mu, sigma = self.fc_mu(y), F.softplus(self.fc_sigma(y))
        dist = torch.distributions.Normal(mu, sigma)
        
        return dist, mu, sigma 
