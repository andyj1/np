#!/usr/bin/env python3

import torch
import torch.nn as nn

import utils.np_utils as np_utils
from botorch.posteriors.posterior import Posterior
from gpytorch import settings
from utils.anp_modules import Decoder, DeterministicEncoder, LatentEncoder

# import torch.optim as optim
# from torch.nn import functional as F
# from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
# from botorch.models.utils import add_output_dim
# from botorch.posteriors.gpytorch import GPyTorchPosterior
# from gpytorch.distributions import (MultitaskMultivariateNormal,MultivariateNormal)
# from gpytorch.lazy import lazify
# from collections import OrderedDict
# from contextlib import ExitStack
# from typing import Any, List, Optional

''' avoids Cholesky decomposition for covariance matrices '''
# 1. covar_root_decomposition: decomposition using low-rank approx using th eLanczos algorithm
# 2. log_prob: computed using a modified conjugate gradients algorithm
# 3. solves: computes positive-definite matrices with preconditioned conjugate gradients
settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ANP(nn.Module):
    def __init__(self, cfg):
        super(ANP, self).__init__()
        num_hidden = cfg['anp']['hidden_dim'] # for encoder and decoder hidden layers
        input_dim = cfg['anp']['input_dim']
        out_dim = cfg['anp']['output_dim']
        
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim=input_dim)    # for z
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden)   # for r
        self.decoder = Decoder(num_hidden)
        self.BCELoss = nn.BCELoss()
        
        self.num_outputs = out_dim
        
        self.r = None
        self.z = None
    
    # training (and evaluating) 
    # same as *fit.py* in custom
    def forward(self, context_x, context_y, target_x, target_y=None):
        
        # add batch size dimension at 0
        context_x = context_x.unsqueeze(0)
        context_y = context_y.unsqueeze(0)
        target_x = target_x.unsqueeze(0)
        target_y = target_y.unsqueeze(0)
        
        # =======save contexts for make_np_posterior============
        self.context_x = context_x
        self.context_y = context_y
        
        # print('context_x, context_y, target_x, target_y ', context_x.shape, context_y.shape, target_x.shape, target_y.shape)
        num_targets = target_x.size(1)

        # returns [mu,sigma] for the input data and [reparameterized sample from that distribution]
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        
        # For training, we need prior AND posterior
        posterior_mu, posterior_var = None, None
        if target_y is not None: # training mode
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            z = posterior
        
        # otherwise for testing, only need prior z (from context)
        else:
            z = prior

        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]
        
        # returns attention query for the encoder input after cross-attention
        r = self.deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]
        
        # prediction of target y given r, z, target_x
        dist, mu, sigma = self.decoder(r, z, target_x)
        
        # For Training
        if target_y is not None:
            log_p = dist.log_prob(target_y)
            posterior = self.latent_encoder(target_x, target_y)
            
            # get log probability
            bce_loss = self.BCELoss(torch.sigmoid(mu), target_y)
            
            # get KL divergence between prior and posterior
            q = torch.distributions.normal.Normal(loc=prior_mu, scale=prior_var)
            p = torch.distributions.normal.Normal(loc=posterior_mu, scale=posterior_var)
            kl = np_utils.kl_div(q, p)
            
            # maximize prob and minimize KL divergence
            loss = bce_loss + kl
            
        # For Generation (testing)
        else:
            log_p = None
            kl = None
            loss = None
            # print('[ANP] Test mode:', end='')
        
        # update r and z for the model
        self.r = r
        self.z = z
        
        return mu, sigma, log_p, kl, loss
    
    
    # posterior generation assuming context x and y already formed r and z
    # returns single-output mvn Posterior class
    # this is called in analytic_np - base class for the acquisition functions
    def make_anp_posterior(self, target_x) -> Posterior:
        target_x = target_x.permute(1,0,2)
        assert self.r.shape[1] == self.z.shape[1] == target_x.shape[1], \
            f'r:{self.r.shape}, z:{self.z.shape}, target_x:{target_x.shape}'
            
        # print(f'[INFO] decoder forwarding... r:{self.r.shape},z:{self.z.shape},target_x:{target_x.shape}')
        dist, _, _ =  self.decoder(self.r, self.z, target_x)
        return dist
        
        