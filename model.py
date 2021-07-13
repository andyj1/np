#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from modules.latent_encoder import LatentEncoder
from modules.deterministic_encoder import DeterministicEncoder
from modules.decoder import Decoder

class AttentiveNP(nn.Module):

    def __init__(self, cfg, device='cuda'):
        super(AttentiveNP, self).__init__()
        self.device = device
        
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
        
        # other approach: reparameterize?
        z_samples = z_samples.unsqueeze(1).repeat(1, target_size, 1)
        
        if self.use_deter_path:
            representation = self.encoder(context_x, context_y, target_x) # [B, target_size, z_dim]
        else:
            representation = None
        # print(representation.shape, z_samples.shape, target_x.shape)
            
        # decode
        decoded_dist, mu, sigma = self.decoder(representation, z_samples, target_x)
        
        if self.training and target_y is not None:
            loss, kld, log_prob = utils.compute_loss(posterior_dist, prior_dist, decoded_dist, target_y, sigma)
            
        else:
            log_prob = None
            kld = None
            loss = None
        return mu, sigma, log_prob, kld, loss



class FeedforwardMLP(nn.Module):

    def __init__(self, cfg, device='cuda'):
        super(FeedforwardMLP, self).__init__()
        self.device = device
        
        self.x_dim = cfg['x_dim']
        self.z_dim = cfg['z_dim']
        self.y_dim = cfg['y_dim']
        
        self.layers = net = nn.Sequential(nn.Linear(self.x_dim, self.z_dim),
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
            loss = self.loss_function(mu, target_y)
            kld = 0.
            log_prob = 0.
        else:
            log_prob = None
            kld = None
            loss = None
        return mu, sigma, log_prob, kld, loss
    