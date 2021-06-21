#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.

Note:   this function is adopted from
        from botorch.acquisition.analytic import (ExpectedImprovement, UpperConfidenceBound)
"""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import (convert_to_target_pre_hook,
                                      t_batch_mode_transform)
from torch import Tensor


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Base class for analytic acquisition functions."""

    def __init__(
         self, model: Model, objective: Optional[ScalarizedObjective] = None
    ): # -> None
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            objective: A ScalarizedObjective (optional).
        """
        super().__init__(model=model)
        if objective is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify an objective when using a multi-output model."
                )
        elif not isinstance(objective, ScalarizedObjective):
            raise UnsupportedError(
                "Only objectives of type ScalarizedObjective are supported for "
                "analytic acquisition functions."
            )
        self.objective = objective

    def _get_posterior(self, X: Tensor): # -> Posterior:
        r"""Compute the posterior at the input candidate set X.

        Applies the objective if provided.

        Args:
            X: The input candidate set.

        Returns:
            The posterior at X. If a ScalarizedObjective is defined, this
            posterior can be single-output even if the underlying model is a
            multi-output model.
        """
        # manipulate shape
        # print(f'[analytic_np -> _get_posterior] input shape:', X.shape)
        
        posterior = self.model.make_np_posterior(X)
        if self.objective is not None:
            # Unlike MCAcquisitionObjective (which transform samples), this
            # transforms the posterior
            posterior = self.objective(posterior)
        return posterior

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )

class UpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    """

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ): # -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, target_x: Tensor): # -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set 'target_x'.

        Args:
            target_x: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(target_x)
        if len(target_x.shape) == 4: target_x = target_x.squeeze(0)
        else: target_x = target_x.permute((1,0,2))
        print('target x:', target_x.shape)
        posterior = self._get_posterior(X=target_x)
        '''
            _get_posterior from the abstract class
            returns a GPyTorchPosterior (single-output mvn) class posterior
        '''
        # batch_shape = target_x.shape[:-2]
        # mean = posterior.mean.view(batch_shape)
        # variance = posterior.variance.view(batch_shape)
        
        # modification
        mean = posterior.mean
        variance = posterior.variance        
        
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            return mean + delta
        else:
            return -mean + delta


class ExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic).

    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `EI(x) = E(max(y - best_f, 0)), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.2)
        >>> ei = EI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor): # -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        mean = posterior.mean
        
        # modification
        mean = posterior.mean
        variance = posterior.variance        
        
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        sigma = variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei

class ProbabilityOfImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Probability of Improvement.

    Probability of improvment over the current best observed value, computed
    using the analytic formula under a Normal posterior distribution. Only
    supports the case of q=1. Requires the posterior to be Gaussian. The model
    must be single-outcome.

    `PI(x) = P(y >= best_f), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PI = ProbabilityOfImprovement(model, best_f=0.2)
        >>> pi = PI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome analytic Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(mode10l=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor): # -> Tensor:
        r"""Evaluate the Probability of Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim tensor of Probability of Improvement values at the given
            design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        
        # mean, sigma = posterior.mean, posterior.variance.sqrt()
        
        # modification
        mean = posterior.mean
        sigma = posterior.variance.sqrt()
        
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        return normal.cdf(u)
