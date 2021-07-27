#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import anp_utils
from modules.latent_encoder import LatentEncoder
from modules.deterministic_encoder import DeterministicEncoder
from modules.decoder import Decoder

import numpy as np
class AttentiveNP(nn.Module):

    def __init__(self, cfg, device='cuda'):
        super(AttentiveNP, self).__init__()
        self.device = device
        self.prior_dist = None # updated at every training iteration
        
        self.num_outputs = 1
        
        self.x_dim = cfg['x_dim']
        self.z_dim = cfg['z_dim']
        self.y_dim = cfg['y_dim']
        
        self.mlp_hidden_size_list = cfg['mlp_hidden_size_list'] # [MLP sizes]
        self.use_self_attention = cfg['use_self_attention']     # True
        self.use_deter_path = cfg['use_deter_path']             # True
        self.self_attention_type = cfg['self_attention_type']   # dot
        self.cross_attention_type = cfg['cross_attention_type'] # multihead
        self.cross_attention_rep = cfg['cross_attention_rep']   # mlp
        
        self.latent_encoder = LatentEncoder(input_x_dim=self.x_dim,
                                            input_y_dim=self.y_dim,
                                            hidden_dim_list=self.mlp_hidden_size_list,
                                            latent_dim=self.z_dim,
                                            self_attention_type=self.self_attention_type,
                                            use_self_attn=True,
                                            attention_layers=2)

        self.decoder = Decoder(x_dim=self.x_dim,
                               y_dim=self.y_dim,
                               mid_hidden_dim_list=self.mlp_hidden_size_list,
                               latent_dim=self.z_dim,  # the dim of last axis of sc and z..
                               use_deterministic_path=self.use_deter_path)  # whether use d_path or not will change the size of input)

        self.encoder = DeterministicEncoder(input_x_dim=self.x_dim,
                                            input_y_dim=self.y_dim,
                                            hidden_dim_list=self.mlp_hidden_size_list,
                                            latent_dim=self.z_dim,
                                            self_attention_type=self.self_attention_type,
                                            use_self_attn=self.use_self_attention,
                                            attention_layers=2,
                                            cross_attention_type=self.cross_attention_type,
                                            cross_attention_rep=self.cross_attention_rep,
                                            attention_dropout=0)        
        
    def forward(self, query, target_y=None):
        # - training: from posterior (encoded contexts+targets)
        # - testing:  from prior (encoded contexts only)
        context_x, context_y, target_x = query
        batch_size, target_size, _ = target_x.shape # size of all x (context+target)
        # assume target = target+context here
        
        # latent encoder: z context
        prior_dist, prior_mu, prior_sigma = self.latent_encoder(context_x, context_y)
        if self.training and target_y is not None:    # train mode
            # latent encoder: z target
            posterior_dist, posterior_mu, posterior_sigma = self.latent_encoder(target_x, target_y)
            z_samples = posterior_dist.rsample()
        else:                                           # test mode
            # use latent context information as prior
            z_samples = prior_dist.rsample()
        
        self.prior_dist = prior_dist # update prior dist for acquisition
        
        # other approach: reparameterize?
        z_samples = z_samples.unsqueeze(1).repeat(1, target_size, 1)
        
        if self.use_deter_path:
            representation = self.encoder(context_x, context_y, target_x) # [B, target_size, z_dim]
            self.representation = representation # update representation for acquisition
        else:
            representation = None
        # print('r,z,x:',representation.shape, z_samples.shape, target_x.shape)
        
        # decode
        decoded_dist, mu, sigma = self.decoder(representation, z_samples, target_x)
        
        if self.training and target_y is not None:
            loss, kld, log_prob = anp_utils.compute_loss(posterior_dist, prior_dist, decoded_dist, target_y, sigma)
            
        else:
            log_prob = None
            kld = None
            loss = None
        return mu, sigma, log_prob, kld, loss

    '''
    # for BoTorch acquisition: no longer used
    def make_posterior(self, target_x):
        batch_size, target_size, _ = target_x.shape # size of all x (context+target)
        z_samples = self.prior_dist.rsample()
        z_samples = z_samples.unsqueeze(1).repeat(1, target_size, 1)
        # print('[make anp posterior]', self.representation.mean(dim=1).unsqueeze(1).shape, z_samples.shape, target_x.shape)
        # torch.Size([16, 1000, 256]) torch.Size([16, 1, 256]) torch.Size([10, 1, 2])
        
        decoded_dist, mu, sigma = self.decoder(self.representation.mean(dim=1).unsqueeze(1), z_samples, target_x)
        return decoded_dist
    
    # black box function (refer to main.py): no longer used
    def black_box_function(self, x1, x2):
        x1 = torch.FloatTensor(x1)
        target_x = torch.cat([x1, x2], dim=-1)
        batch_size, target_size, _ = target_x.shape # size of all x (context+target)
        z_samples = self.prior_dist.rsample()
        z_samples = z_samples.unsqueeze(1).repeat(1, target_size, 1)
        _, mu, sigma = self.decoder(self.representation.mean(dim=1).unsqueeze(1), z_samples, target_x)
        return mu.cpu().detach().numpy()
    '''
    
    # required for utility function (acquisition)
    def predict(self, target_x, return_std=True):
        if isinstance(target_x, np.ndarray):
            target_x = torch.from_numpy(target_x).float().to(self.device)
        
        target_size, input_dim = target_x.shape # size of all x (context+target)
        target_x = target_x.repeat(16, 1, 1).expand(16, -1, -1)
        
        z_samples = self.prior_dist.rsample()
        z_samples = z_samples.unsqueeze(1).repeat(1, target_size, 1)
        
        representation = self.representation.mean(dim=1).unsqueeze(1).repeat(1, target_size, 1)
        
        _, mu, sigma = self.decoder(representation, z_samples, target_x)
        # mu = mu[0]
        # sigma = sigma[0]
        
        mu = mu.mean(dim=0)
        sigma = sigma.mean(dim=0)
        
        mu, sigma = mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()
        
        if return_std:
            return mu, sigma
        
        return mu

class FeedforwardMLP(nn.Module):

    def __init__(self, cfg, device='cuda'):
        super(FeedforwardMLP, self).__init__()
        self.device = device
        
        self.num_outputs = 1
        
        self.x_dim = cfg['x_dim']
        self.z_dim = cfg['z_dim']
        self.y_dim = cfg['y_dim']
        
        self.layers = nn.Sequential(nn.Linear(self.x_dim, self.z_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.z_dim, self.z_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.z_dim, self.y_dim*2)
                                          ).to(self.device)
        self.loss_function = nn.MSELoss()
                
    def forward(self, query, target_y=None):
        context_x, context_y, target_x = query
        batch_size, target_size, _ = target_x.shape # size of all x (context+target)
        
        target_y_pred = self.layers(target_x)
        mu, log_sigma = target_y_pred.chunk(chunks=2, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        
        if self.training and target_y is not None:
            loss = self.loss_function(mu, target_y) + sigma.mean()*0.2
            kld = 0.
            log_prob = 0.
        else:
            log_prob = None
            kld = None
            loss = None
        return mu, sigma, log_prob, kld, loss
    
    # for BoTorch acquisition: no longer used
    def make_posterior(self, target_x):
        target_y_pred = self.layers(target_x)
        mu, log_sigma = target_y_pred.chunk(chunks=2, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        return dist
    
    # required for utility function (acquisition)
    def predict(self, target_x, return_std=True):
        if isinstance(target_x, np.ndarray):
            target_x = torch.from_numpy(target_x).float().to(self.device)
        target_y_pred = self.layers(target_x)
        mu, log_sigma = target_y_pred.chunk(chunks=2, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        
        mu, sigma = mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()
        
        if return_std:
            return mu, sigma
        
        return mu