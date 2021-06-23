#!/usr/bin/env python3

# reference: https://github.com/cqql/neural-processes-pytorch
import sys

import torch
import torch.nn as nn
import utils.np_utils as np_utils
from torch.nn import functional as F
from NPModels import np_modules as np_modules

# color codes for text in UNIX shell
CRED    = '\033[91m'
CGREEN  = '\033[92m'
CCYAN   = '\033[93m'
CBLUE   = '\033[94m'
CEND    = '\033[0m'

class NP(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super(NP, self).__init__()
        
        self.x_dim = cfg['np']['x_dim']
        self.h_dim = cfg['np']['h_dim']
        self.r_dim = cfg['np']['r_dim']
        self.z_dim = cfg['np']['z_dim']
        self.y_dim = cfg['np']['y_dim']

        self.encoder = np_modules.Encoder(x_dim=self.x_dim, y_dim=self.y_dim, r_dim=self.r_dim, h_dim=self.h_dim)
        self.latent_encoder = np_modules.LatentEncoder(r_dim=self.r_dim, z_dim=self.z_dim, h_dim=self.h_dim)
        self.decoder = np_modules.Decoder(x_dim=self.x_dim, y_dim=self.y_dim, z_dim=self.z_dim, h_dim=self.h_dim)

        self.latent_dist = None # keep distribution to sample from across training iterations
        self.p, self.q = None, None
        self.device = device
        # self.std_normal = torch.distributions.Normal(0.0, 1.0)
        # self.elbo_samples = int(self.z_dim * 2)
        
        self.num_outputs = self.y_dim # for botorch dependency

    def forward(self, query, target_y=None):
        context_x, context_y, target_x = query
        
        if self.training and target_y is not None:
            all_x = torch.cat((context_x, target_x), dim=1)
            all_y = torch.cat((context_y, target_y), dim=1)    
        # print('[DATA] context:',context_x.shape, context_y.shape, 'target:', target_x.shape, target_y.shape)
        # [num_samples, context_size, dim]
        
        batch_size, num_targets, _ = target_x.shape # batch size = 1
        _, _, y_dim = context_y.shape
        _, num_contexts, _ = context_x.shape
        
        # latent encoding from contexts
        context_repr = self.encoder(context_x, context_y)
        prior, context_mu, context_sigma = self.latent_encoder(context_repr)
        
        ''' sample z (latent representation) '''
        # train
        if self.training and target_y is not None:
            all_repr = self.encoder(all_x, all_y)
            posterior, all_mu, all_sigma = self.latent_encoder(all_repr)
            
            # set distributions for kl divergence ( target (posterior) || context (prior) )
            p = posterior   # target
            q = prior       # context
            
            # sample from encoded distribution using reparam trick
            self.latent_dist = p
            
        # test
        elif not self.training and target_y is None:
            # set distributions accordingly
            p = None
            q = prior
            
            # sample from encoded distribution using reparam tricks
            self.latent_dist = q
            
            # alternative to sample() method
            # z = torch.rand_like(std) * std + mean
        
        self.p = p
        self.q = q
        
        ''' sample z '''
        # generation (context only)
        # mu, sigma dimension: 1
        z_samples = self.latent_dist.rsample() # sample from latent distribution
        
        ''' decode '''        
        context_pred_mu, context_pred_sigma = self.decoder(context_x, z_samples)        
        target_pred_mu, target_pred_sigma = self.decoder(target_x, z_samples)
        
        logits = {}        
        context_pred_logits = np_utils.logits_from_pred_mu(context_pred_mu, batch_size, self.device)
        target_pred_logits = np_utils.logits_from_pred_mu(target_pred_mu, batch_size, self.device)
        logits.update({'context': context_pred_logits, 'target': target_pred_logits})
        
        # distribution for the predicted/generated target parameters
        p_y_pred = torch.distributions.normal.Normal(target_pred_mu, target_pred_sigma)
        
        ''' set distributions and compute loss '''
        if self.training:
            # loss about the distributions from batch images
            # loss = np_utils.compute_loss(p_y_pred, target_y, p, q, logits['target'])
            loss, kl = np_utils.compute_loss(p_y_pred, target_y, p, q)
            return context_mu, logits, p_y_pred, p, q, loss, kl
        else:
            loss = None
            return p_y_pred, target_pred_mu, target_pred_sigma
        
    # posterior generation assuming context x and y already formed r and z
    # returns single-output mvn Posterior class -> 'from botorch.posteriors.posterior import Posterior'
    # this is called in analytic_np - base class for the acquisition functions
    def make_np_posterior(self, target_x): # -> Posterior
        ''' given z samples from prior, get q ( x | z) '''
        # target_x needs to be of [batch_size, num_samples, dim]
        # q (z | context_x, context_y)
        # print('z:', z_samples.shape, '/ target x:',target_x.shape)
        # print('[NP] target shape', target_x.shape) # 1 (batch_size) x 100 (num_samples) x 2 (dimensions)
        
        print('[NP] target x:', target_x.shape, target_x.dtype)
        # test mode
        self.eval()
        z_samples = self.q.rsample() # for test, sample z from prior
        
        pred_mu, pred_sigma = self.decoder(target_x.to(self.device), z_samples)
        # print(f'[NP] Decoder -- input: {target_x[0]}, output: {pred_mu[0]}, {pred_sigma[0]}')
        dist = torch.distributions.normal.Normal(pred_mu, pred_sigma)
        
        return dist
        