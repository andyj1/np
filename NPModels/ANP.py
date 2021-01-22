
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys

from utils.np_utils import log_likelihood, KLD_gaussian, random_split_context_target

from collections import OrderedDict

from botorch.posteriors.gpytorch import GPyTorchPosterior
from typing import Any, Optional, List
from contextlib import ExitStack
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from gpytorch.lazy import lazify
from botorch.models.utils import add_output_dim

import math

from utils.np_utils import kl_div
from utils.modules import LatentEncoder, DeterministicEncoder, Decoder

class ANP(nn.Module):
    def __init__(self, cfg):  # hidden_dim, decoder_dim, z_samples:
        super(ANP, self).__init__()
        num_hidden = cfg['np']['hidden_dim']
        input_dim = cfg['np']['input_dim']
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim=cfg['np']['input_dim'])    # for z
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden)   # for r
        self.decoder = Decoder(num_hidden)
        self.BCELoss = nn.BCELoss()
    
    # training
    def forward(self, context_x, context_y, target_x, target_y=None):
        
        # add batch size dimension at 0
        context_x = context_x.unsqueeze(0)
        context_y = context_y.unsqueeze(0)
        target_x = target_x.unsqueeze(0)
        target_y = target_y.unsqueeze(0)
        
        print('context_x, context_y, target_x, target_y ', context_x.shape, context_y.shape, target_x.shape, target_y.shape)
        num_targets = target_x.size(1)

        # returns [mu,sigma] for the input data and [reparameterized sample from that distribution]
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        
        # For training, update the latent vector
        posterior_mu, posterior_var = None, None
        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            z = posterior
        
        # For Generation keep the prior distribution as the latent vector
        else:
            z = prior

        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]
        
        # returns attention query for the encoder input after cross-attention
        r = self.deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]
        
        # prediction of target y given r, z, target_x
        dist, y_pred, sigma = self.decoder(r, z, target_x)
        
        # For Training
        if target_y is not None:
            # get log probability
            bce_loss = self.BCELoss(torch.sigmoid(y_pred), target_y)
            
            # get KL divergence between prior and posterior
            kl = kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            
            # maximize prob and minimize KL divergence
            loss = bce_loss + kl
        
        # For Generation
        else:
            log_p = None
            kl = None
            loss = None
        
        print('z:', z.shape, 'r:', r.shape, 'y_pred:',y_pred.shape, 'loss:', loss.item())
        
        return y_pred, sigma, kl, loss

    # evaluate
    # def evaluate(self, context_x, context_y, target_x, None):
        
        