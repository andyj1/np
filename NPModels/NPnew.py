#!/usr/bin/env python3

# reference: https://github.com/cqql/neural-processes-pytorch
import sys

import torch
import torch.nn as nn
import utils.np_utils as np_utils
from torch.nn import functional as F
from NPModels import np_modules

class NP(nn.Module):
    def __init__(self, cfg):
        super(NP, self).__init__()
        
        self.x_dim = cfg['np']['input_dim']  # x
        self.y_dim = cfg['np']['output_dim'] # y
        self.r_dim = cfg['np']['repr_dim']   # r
        self.z_dim = cfg['np']['latent_dim'] # z
        self.h_dim = cfg['np']['hidden_dim'] # h

        self.r_encoder = np_modules.Encoder(self.x_dim, self.y_dim, self.r_dim, self.h_dim)
        self.z_encoder = np_modules.LatentEncoder(self.r_dim, self.z_dim, self.h_dim)
        self.decoder = np_modules.Decoder(self.x_dim, self.y_dim, self.z_dim, self.h_dim)
        
        self.BCELoss = nn.BCELoss(reduction='mean')
        self.num_outputs = self.y_dim
        
        self.r = None
        self.z = None
        
        self.std_normal = torch.distributions.Normal(0.0, 1.0)
        self.elbo_samples = int(self.z_dim * 2)

    def forward(self, query, target_y=None):
        
        context_x, context_y, target_x = query
        
        # add batch size dimension at 0
        all_x = torch.cat([context_x, target_x], dim=0) # concat along dim 0 w/o considering batch size
        all_y = torch.cat([context_y, target_y], dim=0)
        context_x = context_x.unsqueeze(0)
        context_y = context_y.unsqueeze(0)
        target_x = target_x.unsqueeze(0)
        target_y = target_y.unsqueeze(0)
        
        batch_size, num_context, x_dim = context_x.size()
        _, num_target, _ = target_x.size()
        _, _, y_dim = context_y.size()

        # training mode
        if target_y is not None:
            # q (z | x, y)
            r_all = self.r_encoder(all_x, all_y)
            mu_all, sigma_all = self.z_encoder(r_all)
            
            # q (z | context_x, context_y)
            r_context = self.r_encoder(context_x, context_y)
            mu_context, sigma_context = self.z_encoder(r_context)
            
            # kl divergence
            posterior = torch.distributions.normal.Normal(loc=mu_all, scale=sigma_all)
            prior = torch.distributions.normal.Normal(loc=mu_context, scale=sigma_context)
            kl = np_utils.kl_div(posterior, prior)
            
            # estimate the log-likelihood part of the ELBO by sampling z from s_c
            sc = torch.distributions.normal.Normal(mu_all, sigma_all)
            z_sample = sc.sample((self.elbo_samples,))
            # z_sample = z_sample.unsqueeze(1).expand(-1, num_target, -1)    
            
            dist_target, mu_target, sigma_target = self.decoder(target_x, z_sample)
            log_p = np_utils.log_likelihood(mu_target, sigma_target, target_y)
            
            # print('kl:',kl, 'log_p:',log_p)
            
            loss = log_p - kl
        '''
        # testing mode: not explicitly called; called within acquisition function: optimize()
        else:
            # q (z | context_x, context_y)
            r_context = self.r_encoder(context_x, context_y)
            mu_context, sigma_context = self.z_encoder(r_context)
            
            # prior distribution for z
            sc = torch.distributions.Normal(mu_context, sigma_context)
            z_sample = sc.sample()
            
            # reparameterize z for backpropagation through mu and sigma
            z_sample = np_utils.reparametrize(z_sample, num_target)
            
            print('[ELBO] x target shape:',target_x.shape, 'z shape:',z_sample.shape)
            # conditional decoder for all targets
            dist_target, mu_target, sigma_target = self.decoder(target_x, z_sample)
            
            log_p = np_utils.log_likelihood(mu_target, sigma_target, target_x)
            
            kl = None
            loss = None
        '''
        # update r and z for the model
        self.context_x = context_x
        self.context_y = context_y
        self.z_sample = z_sample
        self.r = r_context
        self.sc = sc
         
        return mu_target, sigma_target, dist_target, kl, loss

    # posterior generation assuming context x and y already formed r and z
    # returns single-output mvn Posterior class -> 'from botorch.posteriors.posterior import Posterior'
    # this is called in analytic_np - base class for the acquisition functions
    def make_np_posterior(self, target_x): # -> Posterior
        target_x = target_x.permute(1,0,2)
        self.decoder.eval()
        
        # q (z | context_x, context_y)
        self.r_context = self.r_encoder(self.context_x, self.context_y)
        mu_context, sigma_context = self.z_encoder(self.r_context)

        # since conditional prior is intractable,
        # reparameterize z for backpropagation through mu and sigma
        z = self.std_normal.sample() * sigma_context + mu_context
        self.z = z.unsqueeze(0)
        
        # print('[FORWARD] x target shape:',target_x.shape, 'z shape:',self.z_sample.shape)
        
        # conditional decoder for all targets
        dist_target, mu_target, sigma_target = self.decoder(target_x, self.z)
        # print('mu_target:',mu_target.shape,'sigma_target:',sigma_target.shape)
        return dist_target
        