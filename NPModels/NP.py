#!/usr/bin/env python3

# reference: https://github.com/cqql/neural-processes-pytorch
import sys

import torch
import torch.nn as nn
import utils.np_utils as np_utils
from torch.nn import functional as F
from utils.np_modules import Decoder, DeterministicEncoder, LatentEncoder

class NP(nn.Module):
    def __init__(self, cfg):
        super(NP, self).__init__()
        
        self.x_dim = cfg['np']['input_dim']  # x
        self.y_dim = cfg['np']['output_dim'] # y
        self.r_dim = cfg['np']['repr_dim']   # r
        self.z_dim = cfg['np']['latent_dim'] # z

        self.representation_encoder = DeterministicEncoder(self.x_dim, self.y_dim, self.r_dim)
        self.latent_encoder = LatentEncoder(self.r_dim, self.z_dim)
        self.decoder = Decoder(self.x_dim, self.y_dim, self.z_dim)
                
        self.BCELoss = nn.BCELoss(reduction='mean')
        self.num_outputs = self.y_dim
        
        self.r = None
        self.z = None
        
        self.std_normal = torch.distributions.Normal(0.0, 1.0)
        self.elbo_samples = int(self.z_dim * 2)

    def forward(self, x_context, y_context, x_target, y_target = None):
        
        # add batch size dimension at 0
        x = torch.cat([x_context, x_target], dim=0)
        y = torch.cat([y_context, y_target], dim=0)
        x_context = x_context.unsqueeze(0)
        y_context = y_context.unsqueeze(0)
        x_target = x_target.unsqueeze(0)
        y_target = y_target.unsqueeze(0)
        
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        # training mode
        if y_target is not None:
            # q (z | x, y)
            r_all = self.representation_encoder(x, y)
            mu_all, sigma_all = self.latent_encoder(r_all)
            
            # q (z | x_context, y_context)
            r_context = self.representation_encoder(x_context, y_context)
            mu_context, sigma_context = self.latent_encoder(r_context)
            
            # kl divergence
            posterior = torch.distributions.normal.Normal(loc=mu_all, scale=sigma_all)
            prior = torch.distributions.normal.Normal(loc=mu_context, scale=sigma_context)
            kl = np_utils.kl_div(posterior, prior)
            
            # estimate the log-likelihood part of the ELBO by sampling z from s_c
            sc = torch.distributions.normal.Normal(mu_all, sigma_all)
            z_sample = sc.sample((self.elbo_samples,))
            # z_sample = z_sample.unsqueeze(1).expand(-1, num_target, -1)    
            
            dist_target, mu_target, sigma_target = self.decoder(x_target, z_sample)
            log_p = np_utils.log_likelihood(mu_target, sigma_target, y_target)
            
            print('kl:',kl, 'log_p:',log_p)
            
            loss = log_p - kl
        '''
        # testing mode: not explicitly called; called within acquisition function: optimize()
        else:
            # q (z | x_context, y_context)
            r_context = self.representation_encoder(x_context, y_context)
            mu_context, sigma_context = self.latent_encoder(r_context)
            
            # prior distribution for z
            sc = torch.distributions.Normal(mu_context, sigma_context)
            z_sample = sc.sample()
            
            # reparameterize z for backpropagation through mu and sigma
            z_sample = np_utils.reparametrize(z_sample, num_target)
            
            print('[ELBO] x target shape:',x_target.shape, 'z shape:',z_sample.shape)
            # conditional decoder for all targets
            dist_target, mu_target, sigma_target = self.decoder(x_target, z_sample)
            
            log_p = np_utils.log_likelihood(mu_target, sigma_target, x_target)
            
            kl = None
            loss = None
        '''
        # update r and z for the model
        self.x_context = x_context
        self.y_context = y_context
        self.z_sample = z_sample
        self.r = r_context
        self.sc = sc
         
        return mu_target, sigma_target, dist_target, kl, loss

    # posterior generation assuming context x and y already formed r and z
    # returns single-output mvn Posterior class -> 'from botorch.posteriors.posterior import Posterior'
    # this is called in analytic_np - base class for the acquisition functions
    def make_np_posterior(self, x_target): # -> Posterior
        x_target = x_target.permute(1,0,2)
        self.decoder.eval()
        
        # q (z | x_context, y_context)
        self.r_context = self.representation_encoder(self.x_context, self.y_context)
        mu_context, sigma_context = self.latent_encoder(self.r_context)

        # since conditional prior is intractable,
        # reparameterize z for backpropagation through mu and sigma
        z = self.std_normal.sample() * sigma_context + mu_context
        self.z = z.unsqueeze(0)
        
        # print('[FORWARD] x target shape:',x_target.shape, 'z shape:',self.z_sample.shape)
        
        # conditional decoder for all targets
        dist_target, mu_target, sigma_target = self.decoder(x_target, self.z)
        # print('mu_target:',mu_target.shape,'sigma_target:',sigma_target.shape)
        return dist_target
        