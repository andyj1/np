#!/usr/bin/env python3

import os
import sys

import numpy as np
import torch
from torch.distributions.kl import kl_divergence

# for kullback-leibler divergence computation
sys.setrecursionlimit(1500)

CRED    = '\033[91m'
CBLUE   = '\033[94m'
CEND    = '\033[0m'

def log_likelihood(mu, sigma, target):
    target = target.squeeze(0)
    target = target.unsqueeze(dim=1)
    mu = mu.unsqueeze(dim=0)
    sigma = sigma.unsqueeze(dim=0)
    '''
    [LOG LIKELIHOOD] target: torch.Size([num_target, 1, 1]) 
                        mu: torch.Size([1, num_target, z_dim*2, 1]) 
                        sigma: torch.Size([1, num_target, z_dim*2, 1])
    '''
    # print('target:', target.shape, 'mu:',mu.shape, 'sigma:',sigma.shape)
    
    # norm = torch.distributions.Normal(mu.squeeze(0), sigma.squeeze(0))
    # return norm.log_prob(target).sum(dim=0).mean()

    # equation: https://www.statlect.com/glossary/log-likelihood
    return -(target - mu)**2 / (2 * sigma**2) - torch.log(sigma)

def kl_div(q_target, p_context):
    ''' example
    posterior = torch.distributions.normal.Normal(loc=mu, scale=sigma)
    prior = torch.distributions.normal.Normal(loc=mu, scale=sigma)
    kl_div(posterior, prior) := KLD(posterior || prior)
    '''
    return kl_divergence(q_target, p_context).mean(dim=0).sum()

def random_split_context_target(x, y, n_context):
    # cpu-only
    # ind = np.arange(x.shape[0])
    # mask = np.random.choice(ind, size=n_context, replace=False)
    # x = x.cpu()
    # y = y.cpu()
    # context_x, context_y, target_x, target_y = [x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)]
    
    # alternative
    # context_idx = random.sample(range(train_X.shape[0]), cfg['np']['num_context'])
    # target_idx = np.delete(range(train_X.shape[0]), context_idx)
    # x_context, x_target = train_X[context_idx], train_X[target_idx]
    # y_context, y_target = train_Y[context_idx], train_Y[target_idx]
    
    # randomly select context and target indices
    indices = torch.randperm(x.shape[0])
    context_indices = indices[:n_context]
    target_indices = indices[n_context:]
    
    # split into context and target
    context_x, context_y = x[context_indices,:], y[context_indices,:]
    target_x, target_y = x[target_indices,:], y[target_indices,:]
    
    return context_x, context_y, target_x, target_y

# reparam trick for tractability in randomness in the input
# def reparametrize(mu, logvar, num_target):
#     std = torch.exp(0.5 * logvar)
#     eps = torch.randn_like(std)
#     z_sample = eps.mul(std).add_(mu)
#     z_sample = z_sample.unsqueeze(1).expand(-1, num_target, -1)
#     return z_sample

# def compute_loss(p_y_pred, target_y, q_target, q_context, target_pred_logits):
def compute_loss(p_y_pred, target_y, p, q):
    # for images
    # bce = nn.BCELoss(reduction='mean')
    # bce_loss = torch.nn.functional.binary_cross_entropy(target_pred_logits, target_y)
    
    # for continuous (signal) data
    log_likelihood = p_y_pred.log_prob(target_y).mean(dim=0).sum()
    kl = torch.distributions.kl.kl_divergence(p, q).mean(dim=0).sum()
    orig_loss = log_likelihood - kl
    orig_loss *= -1
    # print(' loss...')
    # print('log likelihood:', log_likelihood, f'kl:{kl:2.9f}')
    
    loss = orig_loss
    
    # print(f' loss: {CRED}{loss.item():10.5f}{CEND} / kl: {CBLUE}{kl.item():10.5f}{CEND}')

    return loss, kl

# https://discuss.pytorch.org/t/vae-example-reparametrize/35843
# logVariance = log($\sigma^2$) = 2 * log(sigma)
# logStdDev = 2 * log(sigma) / 2 = 0.5 * 2 * log(sigma) = log(sigma)
def logvar_to_std(logvar):
    logstd = (1/2) * logvar
    std = torch.exp(logstd)
    
    # as per original deepmind code (in LatentEncoder)
    # used in LatentEncoder
    std = 0.1 + 0.9 * torch.sigmoid(logvar)
    
    # === reparameterization trick ===
    # "Empirical Evaluation of Neural Process Objectives" and "Attentive Neural Processes"
    # reparameterization trick
    # sigma = 0.1 + 0.9 * sigma
    # z_sample = self.std_normal.sample() * sigma + mu
    # z_sample = z_sample.unsqueeze(0)
    
    # uniform distribution
    # z_samples = torch.rand_like(std) * std + mu
    # normal distribution
    return std


def logstd_to_std(logstd):
    # used in Decoder
    activation = torch.nn.Sigmoid()
    bounded_std = 0.1 + 0.9 * activation(logstd)
    return bounded_std

def logits_from_pred_mu(pred_mu, batch_size, device='cuda'):
    if not pred_mu.is_cuda: 
        pred_mu = pred_mu.to(device)
    logits = torch.tensor([]).to(device)
    for i in range(batch_size):
        index = torch.tensor([i]).to(device)
        indexed_pred_mu = torch.index_select(pred_mu, 0, index)
        new_logits = torch.sigmoid(indexed_pred_mu)
        logits = torch.cat([logits, new_logits], dim=0)
    return logits

def clip_tensor(input:torch.Tensor = torch.tensor([]), min=0, max=1):
    output = torch.clip(input, 0, 1)
    return output