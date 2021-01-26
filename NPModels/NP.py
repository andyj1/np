#!/usr/bin/env python3

import sys
from collections import OrderedDict
from contextlib import ExitStack
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models.utils import add_output_dim
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import settings
from gpytorch.distributions import (MultitaskMultivariateNormal,
                                    MultivariateNormal)
# from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from gpytorch.lazy import lazify
from torch.nn import functional as F
from utils.np_modules import Encoder, Decoder, Repr2Latent
import utils.np_utils as np_utils
from botorch.posteriors.posterior import Posterior


''' avoids Cholesky decomposition for covariance matrices '''
# 1. covar_root_decomposition: decomposition using low-rank approx using th eLanczos algorithm
# 2. log_prob: computed using a modified conjugate gradients algorithm
# 3. solves: computes positive-definite matrices with preconditioned conjugate gradients
settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NP(nn.Module):
    def __init__(self, cfg):
        super(NP, self).__init__()
        self.x_dim = cfg['np']['input_dim']
        self.y_dim = cfg['np']['output_dim']
        self.r_dim = cfg['np']['repr_dim']
        self.z_dim = cfg['np']['latent_dim'] 
        self.h_dim = cfg['np']['hidden_dim'] # for encoder and decoder hidden layers

        self.encoder = Encoder(self.x_dim, self.y_dim, self.h_dim, self.r_dim)
        self.r_to_z = Repr2Latent(self.r_dim, self.z_dim)
        self.decoder = Decoder(self.x_dim, self.z_dim, self.h_dim, self.y_dim)
        
        self.BCELoss = nn.BCELoss()
        self.num_outputs = self.y_dim
        
        self.r = None
        self.z = None

    def aggregate(self, r_i):
            """
            Aggregates representations for every (x_i, y_i) pair into a single
            representation.
            Parameters
            ----------
            r_i : torch.Tensor
                Shape (batch_size, num_points, r_dim)
            """
            return torch.mean(r_i, dim=1)
    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.encoder(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_z(r)

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
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            posterior = torch.distributions.Normal(mu_target, sigma_target)
            prior = torch.distributions.Normal(mu_context, sigma_context)
            
            # find z
            # TODO: need to reparameterize and make this samples
            # z = posterior.unsqueeze(1).repeat(1, num_target, 1)
            
            # find r
            batch_size, num_target, _ = x_target.size()
            # Flatten tensors, as encoder expects one dimensional inputs
            # TODO
            
            x_flat = x_context.view(batch_size * num_target, self.x_dim)
            y_flat = y_context.contiguous().view(batch_size * num_target, self.y_dim)
            # Encode each point into a representation r_i
            r_i_flat = self.encoder(x_flat, y_flat)
            # Reshape tensors into batches
            r_i = r_i_flat.view(batch_size, num_target, self.r_dim)
            # Aggregate representations r_i into a single representation r
            r = self.aggregate(r_i)
            
            
            kl = np_utils.kl_div(prior, posterior)
            
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
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            prior = torch.distributions.Normal(mu_context, sigma_context)
            z_sample = prior.rsample()
            
            # find z
            z = z_sample.unsqueeze(1).repeat(1, num_target, 1)
            
            
            # find r
            batch_size, num_target, _ = x_target.size()
            # Flatten tensors, as encoder expects one dimensional inputs
            x_flat = x_context.view(batch_size * num_target, self.x_dim)
            y_flat = y_context.contiguous().view(batch_size * num_target, self.y_dim)
            # Encode each point into a representation r_i
            r_i_flat = self.encoder(x_flat, y_flat)
            # Reshape tensors into batches
            r_i = r_i_flat.view(batch_size, num_target, self.r_dim)
            # Aggregate representations r_i into a single representation r
            r = self.aggregate(r_i)
            
            # Predict target points based on context
            y_pred_dist, y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample)
            p_y_pred = torch.distributions.Normal(y_pred_mu, y_pred_sigma)
            
            log_p = None
            kl = None
            loss = None
           
        # update r and z for the model
        self.r = r
        self.z = z
         
        return y_pred_mu, y_pred_sigma, log_p, kl, loss

    # posterior generation assuming context x and y already formed r and z
    # returns single-output mvn Posterior class
    # this is called in analytic_np - base class for the acquisition functions
    def make_np_posterior(self, target_x) -> Posterior:
        target_x = target_x.permute(1,0,2)
        assert self.r.shape[1] == self.z.shape[1] == target_x.shape[1], \
            f'r:{self.r.shape}, z:{self.z.shape}, target_x:{target_x.shape}'
            
        # print(f'[INFO] decoder forwarding... r:{self.r.shape},z:{self.z.shape},target_x:{target_x.shape}')
        dist, _, _ =  self.decoder(self.r, self.z, target_x)
        return dist
        