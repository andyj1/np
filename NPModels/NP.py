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
    def __init__(self, cfg, device):
        super(NP, self).__init__()
        
        self.x_dim = cfg['np']['x_dim']
        self.y_dim = cfg['np']['y_dim']
        self.r_dim = cfg['np']['r_dim']
        self.z_dim = cfg['np']['z_dim']
        self.h_dim = cfg['np']['h_dim']

        self.encoder = np_modules.Encoder(x_dim=self.x_dim, y_dim=self.y_dim, r_dim=self.r_dim, h_dim=self.h_dim)
        self.latent_encoder = np_modules.LatentEncoder(r_dim=self.r_dim, z_dim=self.z_dim, h_dim=self.h_dim)
        self.decoder = np_modules.Decoder(x_dim=self.x_dim, y_dim=self.y_dim, z_dim=self.z_dim, h_dim=self.h_dim)

        self.BCELoss = nn.BCELoss(reduction='mean')
        
        self.z_samples = None
        self.device = device
        
        self.std_normal = torch.distributions.Normal(0.0, 1.0)
        self.num_outputs = self.y_dim # for botorch dependency
        self.elbo_samples = int(self.z_dim * 2)

    def forward(self, query, target_y=None):
        context_x, context_y, target_x = query
        context_x = context_x.unsqueeze(0)
        context_y = context_y.unsqueeze(0)
        target_x = target_x.unsqueeze(0)
        if target_y is not None:
            target_y = target_y.unsqueeze(0)

        if self.training and target_y is not None:
            all_x = torch.cat((context_x, target_x), dim=1)
            all_y = torch.cat((context_y, target_y), dim=1)    
        # print('[DATA] context:',context_x.shape, context_y.shape, 'target:', target_x.shape, target_y.shape)
        # [num_samples, context_size, dim]
        
        batch_size, num_targets, x_dim = target_x.shape
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
            
            # set distributions for kl divegence ( target (posterior) || context (prior) )
            q_target = posterior
            q_context = prior
            
            # sample from encoded distribution using reparam trick
            z_samples = q_target.rsample() # sampled from context+target encoded vector
        # test
        elif not self.training and target_y is None:
            # set distributions accordingly
            q_target = None
            q_context = prior
            
            # sample from encoded distribution using reparam tricks
            z_samples = q_context.rsample()
            
            # alternative to sample() method
            # z = torch.rand_like(std) * std + mean
        
        # store z samples for acquistion
        self.z_samples = z_samples
        
        ''' decode '''
        # generation (context only)
        # mu, sigma dimension: 1 
        context_pred_mu, context_pred_sigma = self.decoder(context_x, z_samples)
        context_pred_logits = np_utils.logits_from_pred_mu(context_pred_mu, batch_size, self.device)
        
        target_pred_mu, target_pred_sigma = self.decoder(target_x, z_samples)
        target_pred_logits = np_utils.logits_from_pred_mu(target_pred_mu, batch_size, self.device)
        
        logits = {}        
        logits.update({'context': context_pred_logits, 'target': target_pred_logits})
        
        # distribution for the predicted/generated target parameters    
        p_y_pred = torch.distributions.normal.Normal(target_pred_mu, target_pred_sigma)
        
        ''' set distributions and compute loss '''
        if self.training:
            # loss about the distributions from batch images
            loss = np_utils.compute_loss(p_y_pred, target_y, q_target, q_context, logits['target'])
        else:
            loss = None
        
        return context_mu, logits, p_y_pred, q_target, q_context, loss    

    # posterior generation assuming context x and y already formed r and z
    # returns single-output mvn Posterior class -> 'from botorch.posteriors.posterior import Posterior'
    # this is called in analytic_np - base class for the acquisition functions
    def make_np_posterior(self, target_x): # -> Posterior
        # target_x needs to be of [batch_size, num_samples, dim]
        
        # test mode
        # q (z | context_x, context_y)
        # given z samples from prior, get q ( x | z)
        print('z:', self.z_samples.shape, '/ target x:',target_x.shape)
        pred_mu, pred_sigma = self.decoder(target_x.to(self.device), self.z_samples)
        dist = torch.distributions.normal.Normal(pred_mu, pred_sigma)
        print('mu_target:',pred_mu.shape,'sigma_target:',pred_sigma.shape)
        return dist
        