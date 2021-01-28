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
from utils.np_utils import kl_div, log_likelihood, random_split_context_target

''' avoids Cholesky decomposition for covariance matrices '''
# 1. covar_root_decomposition: decomposition using low-rank approx using th eLanczos algorithm
# 2. log_prob: computed using a modified conjugate gradients algorithm
# 3. solves: computes positive-definite matrices with preconditioned conjugate gradients
settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NP(nn.Module):
    def __init__(self, cfg):
        super(NP, self).__init__()
        in_dim = cfg['np']['input_dim']
        out_dim = cfg['np']['output_dim']
        
        self.in_dim = in_dim
        self.num_outputs = out_dim
        self.r_dim = 1
        self.z_dim = 1
        self.z_samples = cfg['np']['hidden_dim']

        self.mu_context = torch.Tensor([0.0 for _ in range(self.r_dim)])
        self.sigma_context = torch.Tensor([0.0 for _ in range(self.r_dim)])

        # for data (Xc, Yc) --> representation (r)
        self.encoder = nn.Sequential(OrderedDict([
            ('hidden1', nn.Linear(in_dim + out_dim, cfg['hidden_dim'])),
            ('relu1', nn.ReLU()),
            ('hidden2', nn.Linear(cfg['np']['hidden_dim'], self.r_dim))
        ]))
        # for representation (r) --> latent vector (z)
        self.r_to_z_mu = nn.Linear(self.r_dim, self.z_dim) # quick fix: in_dim - 1 (need to verify)
        self.r_to_z_std = nn.Linear(self.r_dim, self.z_dim)

        # for input(x) & latent vector(z) --> output(y)
        self.decoder_hidden = nn.Sequential(OrderedDict([
            ('decoder1', nn.Linear(in_dim + self.z_dim, cfg['decoder_dim'])),
            ('sigmoid1', nn.Sigmoid()),
            # ('decoder2', nn.Linear(cfg['decoder_dim'], cfg['decoder_dim']))
        ]))
        self.decoder_mu = nn.Linear(cfg['decoder_dim'], out_dim)
        self.decoder_std = nn.Linear(cfg['decoder_dim'], out_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)

    # data to representations, aggregated
    def data_to_r(self, x, y):
        x_y = torch.cat([x, y], dim=1)
        r_i = self.encoder(x_y)

        # mean aggregate
        r = r_i.mean(dim=0)
        return r

    # representation to latent vector
    def r_to_z(self, r):
        mean = self.r_to_z_mu(r)
        log_var = self.r_to_z_std(r)
        return mean, F.softplus(log_var)

    # reparam trick for tractability in randomness in the input
    def reparametrize(self, mu, std, n):
        eps = torch.autograd.Variable(std.data.new(n, self.z_dim).normal_())
        z = eps.mul(std).add_(mu)
        return z

    # decoder
    def decoder(self, x, z):
        # z = z.unsqueeze(-1).expand(z.size(0), z.size(1), x_pred.size(0)).transpose(1, 2)
        # x_pred = x_pred.unsqueeze(0).expand(z.size(0), x_pred.size(0), x_pred.size(1))

        # Concatenate to a tensor of shape [len(z), len(x), x_dim + z_dim]
        x = x.reshape((-1, self.in_dim))
        z = z.reshape((-1, self.z_dim))
        x = x.unsqueeze(0).expand(z.shape[0], -1, -1)
        z = z.unsqueeze(0).expand(x.shape[1], -1, -1).transpose(0, 1)

        x_z = torch.cat([x, z], dim=-1)

        decoded = self.decoder_hidden(x_z)
        mu = self.decoder_mu(decoded)
        std = self.decoder_std(decoded)

        mu = mu.squeeze(-1)
        std = std.squeeze(-1)
        std = F.softplus(std)
        return mu, std

    # def forward(self, x_context, y_context, x_target, y_target = None):
    def forward(self, X, y_target = None):
        # x_context, y_context, x_target = torch.split(X, 3, dim = -1)
        # x_context, y_context, x_target = X
        # in_dim = x_target.shape[-1]
        if self.training :
            x_context, y_context, x_target = X
            in_dim = x_target.shape[-1]

            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.r_to_z(self.data_to_r(x_target, y_target))
            mu_context, sigma_context = self.r_to_z(self.data_to_r(x_context, y_context))
            # Sample from encoded distribution using reparameterization trick
            q_target = torch.distributions.Normal(mu_target, sigma_target)
            q_context = torch.distributions.Normal(mu_context, sigma_context)

            z_sample = q_target.rsample([self.z_samples, self.z_dim])
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample)
            # p_y_pred = torch.distributions.Normal(y_pred_mu, y_pred_sigma)

            self.mu_context = mu_context
            self.sigma_context = sigma_context

            return y_pred_mu, y_pred_sigma, q_target, q_context
        else:
            # At testing time, encode only context
            in_dim = X.shape[-1]
            # FIXME
            # mu_context, sigma_context = self.data_to_r(x_context, y_context)
            # Sample from distribution based on context
            q_context = torch.distributions.Normal(self.mu_context, self.sigma_context)
            z_sample = q_context.rsample([self.z_samples, in_dim])
            # Predict target points based on context
            # y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample)
            y_pred_mu, y_pred_sigma = self.decoder(X, z_sample)
            # p_y_pred = torch.distributions.Normal(y_pred_mu, y_pred_sigma)
            # return y_pred_mu, y_pred_sigma
            return MultivariateNormal(y_pred_mu, y_pred_sigma**2)

    '''
    def forward(self, x_context, y_context, x_target, y_target):
        # make x, y stack
        x_all = torch.cat([x_context, x_target], dim = 0)
        y_all = torch.cat([y_context, y_target], dim = 0)

        # all data -> global repr
        r_all = self.data_to_r(x_all, y_all)
        z_all_mu, z_all_std = self.r_to_z(r_all)

        # context data -> context repr
        r = self.data_to_r(x_context, y_context)    # data -> repr
        # print('after data_to_r, r shape:',r.shape)

        z_mean, z_std = self.r_to_z(r)   # repr -> latent z

        # reparameterize
        # Estimate the log-likelihood part of the ELBO by sampling z from q(z | x, y)
        zs = self.reparametrize(z_all_mu, z_all_std, self.z_samples)

        # decoder
        mu, std = self.decoder(x_target, zs)
        # mu, std = self.decoder(x_context, zs)
        return mu, std, z_all_mu, z_all_std, z_mean, z_std

    def elbo(self, x_context, y_context, x_target, y_target):
        mu, std, z_mu_all, z_std_all, z_mean, z_std = self.forward(x_context, y_context, x_target, y_target)

        # Compute the Kullback-Leibler divergence part
        kld = kullback_leiber_divergence(z_mu_all, z_std_all, z_mean, z_std)
        log_llh = log_likelihood(mu, std, y_target).sum(dim=0).mean()
        return log_llh - kld
    '''

    def posterior(
            self,
            X: torch.Tensor,
            output_indices: Optional[List[int]] = None,
            observation_noise: bool = False,
            **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension of the
                feature space and `q` is the number of points considered jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.
            detach_test_caches: If True, detach GPyTorch test caches during
                computation of the posterior. Required for being able to compute
                derivatives with respect to training inputs at test time (used
                e.g. by qNoisyExpectedImprovement). Defaults to `True`.

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices` each. Includes observation noise if
            `observation_noise=True`.
        """
        self.eval()  # make sure model is in eval mode
        detach_test_caches = kwargs.get("detach_test_caches", True)
        with ExitStack() as es:
            es.enter_context(settings.debug(False))
            es.enter_context(settings.fast_pred_var())
            es.enter_context(settings.detach_test_caches(detach_test_caches))
            # insert a dimension for the output dimension
            if self.num_outputs > 1:
                X, output_dim_idx = add_output_dim(
                    X=X, original_batch_shape=self._input_batch_shape
                )
            mvn = self(X)
            mean_x = mvn.mean
            covar_x = mvn.covariance_matrix
            if self.num_outputs > 1:
                output_indices = output_indices or range(self.num_outputs)
                mvns = [
                    MultivariateNormal(
                        mean_x.select(dim=output_dim_idx, index=t),
                        lazify(covar_x.select(dim=output_dim_idx, index=t)),
                    )
                    for t in output_indices
                ]
                mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
        return GPyTorchPosterior(mvn=mvn)
