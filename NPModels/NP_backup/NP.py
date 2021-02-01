#!/usr/bin/env python3

# reference: https://github.com/EmilienDupont/neural-processes
import sys

import torch
import torch.nn as nn
import utils.np_utils as np_utils
from botorch.posteriors.posterior import Posterior
from torch.nn import functional as F
from utils.np_modules import Decoder, DeterministicEncoder, LatentEncoder

class NP(nn.Module):
    def __init__(self, cfg):
        super(NP, self).__init__()
        self.x_dim = cfg['np']['input_dim']  # x
        self.y_dim = cfg['np']['output_dim'] # y
        self.r_dim = cfg['np']['repr_dim']   # r
        self.z_dim = cfg['np']['latent_dim'] # z
        self.h_dim = cfg['np']['hidden_dim'] # for encoder and decoder hidden layers

        self.encoder = DeterministicEncoder(self.x_dim, self.y_dim, self.h_dim, self.r_dim)
        self.latent_encoder = LatentEncoder(self.r_dim, self.z_dim, self.h_dim, self.x_dim, self.y_dim)
        self.decoder = Decoder(self.x_dim, self.z_dim, self.h_dim, self.y_dim)
        
        self.BCELoss = nn.BCELoss(reduction='mean')
        self.num_outputs = self.y_dim
        
        self.r = None
        self.z = None


    def forward(self, x_context, y_context, x_target, y_target = None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.
        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.
        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.
        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        
        # add batch size dimension at 0
        x_context = x_context.unsqueeze(0)
        y_context = y_context.unsqueeze(0)
        x_target = x_target.unsqueeze(0)
        target_y = y_target.unsqueeze(0)
        
        # Infer quantities from tensor dimensions
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        
        # training mode
        if y_target is not None: 
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.latent_encoder(x_target, y_target)
            mu_context, sigma_context = self.latent_encoder(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            posterior = torch.distributions.Normal(mu_target, sigma_target)
            prior = torch.distributions.Normal(mu_context, sigma_context)
            kl = np_utils.kl_div(prior, posterior)
            
            # find z
            z_sample = posterior.rsample()
            
            
            z = z_sample.unsqueeze(1).repeat(1, num_target, 1)

            # Get parameters of output distribution
            y_pred_dist, y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample)
            p_y_pred = torch.distributions.Normal(y_pred_mu, y_pred_sigma)
            log_p = p_y_pred.log_prob(y_target)

            loss = self.BCELoss(torch.sigmoid(y_pred_mu), target_y)
    
        # testing mode
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.latent_encoder(x_context, y_context)
            # Sample from distribution based on context
            prior = torch.distributions.Normal(mu_context, sigma_context)
            z_sample = prior.rsample()
            
            # find z
            z = z_sample.unsqueeze(1).repeat(1, num_target, 1)
                 
            # Predict target points based on context
            y_pred_dist, y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample)
            p_y_pred = torch.distributions.Normal(y_pred_mu, y_pred_sigma)
            
            log_p = None
            kl = None
            loss = None
           
        # update r and z for the model
        self.z_sample = z_sample
         
        return y_pred_mu, y_pred_sigma, log_p, kl, loss

    # posterior generation assuming context x and y already formed r and z
    # returns single-output mvn Posterior class
    # this is called in analytic_np - base class for the acquisition functions
    def make_np_posterior(self, target_x) -> Posterior:
        target_x = target_x.permute(1,0,2)
        self.decoder.eval()
        # print(f'target_x:{target_x}, z_sample:{self.z_sample}')
        
        dist, _, _ =  self.decoder(target_x, self.z_sample)
        return dist
        