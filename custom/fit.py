#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Tools for model fitting.

Note:   this function is customized:
        from botorch.optim.fit import fit_gpytorch_torch
"""

from __future__ import annotations

import logging
import time
import warnings
from copy import deepcopy
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Set,
                    Tuple, Union)

import numpy as np
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning, OptimizationWarning
from botorch.models.converter import (batched_to_model_list,
                                      model_list_to_batched)
from botorch.models.gp_regression import HeteroskedasticSingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.numpy_converter import (TorchAttr, module_to_array,
                                           set_params_with_array)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import (_filter_kwargs, _get_extra_mll_args,
                                 _scipy_objective_and_grad, sample_all_priors)
from gpytorch import settings as gpt_settings
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from scipy.optimize import Bounds, minimize
from torch import Tensor
from torch.nn import Module
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]
TScipyObjective = Callable[
    [np.ndarray, MarginalLogLikelihood, Dict[str, TorchAttr]], Tuple[float, np.ndarray]
]
TModToArray = Callable[
    [Module, Optional[ParameterBounds], Optional[Set[str]]],
    Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]],
]
TArrayToMod = Callable[[Module, np.ndarray, Dict[str, TorchAttr]], Module]


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


FAILED_CONVERSION_MSG = (
    "Failed to convert ModelList to batched model. "
    "Performing joint instead of sequential fitting."
)


def fit_gpytorch_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    r"""Fit hyperparameters of a GPyTorch model.

    On optimizer failures, a new initial condition is sampled from the
    hyperparameter priors and optimization is retried. The maximum number of
    retries can be passed in as a `max_retries` kwarg (default is 5).

    Optimizer functions are in botorch.optim.fit.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        optimizer: The optimizer function.
        kwargs: Arguments passed along to the optimizer function, including
            `max_retries` and `sequential` (controls the fitting of `ModelListGP`
            and `BatchedMultiOutputGPyTorchModel` models) or `approx_mll`
            (whether to use gpytorch's approximate MLL computation).

    Returns:
        MarginalLogLikelihood with optimized parameters.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> fit_gpytorch_model(mll)
    """
    sequential = kwargs.pop("sequential", True)
    max_retries = kwargs.pop("max_retries", 5)
    if isinstance(mll, SumMarginalLogLikelihood) and sequential:
        for mll_ in mll.mlls:
            fit_gpytorch_model(
                mll=mll_, optimizer=optimizer, max_retries=max_retries, **kwargs
            )
        return mll
    elif (
        isinstance(mll.model, BatchedMultiOutputGPyTorchModel)
        and mll.model._num_outputs > 1
        and sequential
    ):
        tf = None
        try:  # check if backwards-conversion is possible
            # remove the outcome transform since the training targets are already
            # transformed and the outcome transform cannot currently be split.
            # TODO: support splitting outcome transforms.
            if hasattr(mll.model, "outcome_transform"):
                tf = mll.model.outcome_transform
                mll.model.outcome_transform = None
            model_list = batched_to_model_list(mll.model)
            model_ = model_list_to_batched(model_list)
            mll_ = SumMarginalLogLikelihood(model_list.likelihood, model_list)
            fit_gpytorch_model(
                mll=mll_,
                optimizer=optimizer,
                sequential=True,
                max_retries=max_retries,
                **kwargs,
            )
            model_ = model_list_to_batched(mll_.model)
            mll.model.load_state_dict(model_.state_dict())
            # setting the transformed inputs is necessary because gpytorch
            # stores the raw training inputs on the ExactGP in the
            # ExactGP.__init__ call. At evaluation time, the test inputs will
            # already be in the transformed space if some transforms have
            # transform_on_eval set to False. ExactGP.__call__ will
            # concatenate the test points with the training inputs. Therefore,
            # it is important to set the ExactGP's train_inputs to also be
            # transformed data using all transforms (including the transforms
            # with transform_on_train set to True).
            mll.train()
            _set_transformed_inputs(mll=mll)
            if tf is not None:
                mll.model.outcome_transform = tf
            return mll.eval()
        # NotImplementedError is omitted since it derives from RuntimeError
        except (UnsupportedError, RuntimeError, AttributeError):
            warnings.warn(FAILED_CONVERSION_MSG, BotorchWarning)
            if tf is not None:
                mll.model.outcome_transform = tf
            return fit_gpytorch_model(
                mll=mll, optimizer=optimizer, sequential=False, max_retries=max_retries
            )
    # retry with random samples from the priors upon failure
    mll.train()
    original_state_dict = deepcopy(mll.model.state_dict())
    retry = 0
    while retry < max_retries:
        with warnings.catch_warnings(record=True) as ws:
            if retry > 0:  # use normal initial conditions on first try
                mll.model.load_state_dict(original_state_dict)
                sample_all_priors(mll.model)
            mll, _ = optimizer(mll, track_iterations=False, **kwargs)
            if not any(issubclass(w.category, OptimizationWarning) for w in ws):
                _set_transformed_inputs(mll=mll)
                mll.eval()
                return mll
            retry += 1
            logging.log(logging.DEBUG, f"Fitting failed on try {retry}.")

    warnings.warn("Fitting failed on all retries.", OptimizationWarning)
    return mll.eval()

def _set_transformed_inputs(mll: MarginalLogLikelihood) -> None:
    r"""Update training inputs with transformed inputs.

    Args:
        mll: The marginal likelihood.
    """
    models = (
        mll.model.models if isinstance(mll, SumMarginalLogLikelihood) else [mll.model]
    )
    for m in models:
        if hasattr(m, "input_transform"):
            X_tf = m.input_transform.preprocess_transform(m.train_inputs[0])
            if not hasattr(m, "set_train_data"):
                raise BotorchError(
                    "fit_gpytorch_model requires that a model has a set_train_data "
                    "method when an input_transform is used."
                )
            m.set_train_data(X_tf, strict=False)
            # TODO: override set_train_data in HeteroskedasticSingleTaskGP to do this
            # automatically
            if isinstance(m, HeteroskedasticSingleTaskGP):
                m.likelihood.noise_covar.noise_model.set_train_data(X_tf, strict=False)

def fit_gpytorch_torch(
    mll: MarginalLogLikelihood,
    bounds: Optional[ParameterBounds] = None,
    optimizer_cls: Optimizer = Adam,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
    approx_mll: bool = True,
    custom_optimizer: Optional[Optimizer] = None,
    display_for_every = 10
):# -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    r"""Fit a gpytorch model by maximizing MLL with a torch optimizer.

    The model and likelihood in mll must already be in train mode.
    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        optimizer_cls: Torch optimizer to use. Must not require a closure.
        options: options for model fitting. Relevant options will be passed to
            the `optimizer_cls`. Additionally, options can include: "disp"
            to specify whether to display model fitting diagnostics and "maxiter"
            to specify the maximum number of iterations.
        track_iterations: Track the function values and wall time for each
            iteration.
        approx_mll: If True, use gpytorch's approximate MLL computation (
            according to the gpytorch defaults based on the training at size).
            Unlike for the deterministic algorithms used in fit_gpytorch_scipy,
            this is not an issue for stochastic optimizers.

    Returns:
        2-element tuple containing
        - mll with parameters optimized in-place.
        - Dictionary with the following key/values:
        "fopt": Best mll value.
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationIteration objects with information on each
        iteration. If track_iterations is False, will be empty.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_torch(mll)
        >>> mll.eval()
    """
    optim_options = options if options is not None else {"maxiter": 100, "disp": True, "lr": 0.05}
    optim_options.update(options or {})
    exclude = optim_options.pop("exclude", None)
    DISPLAY_FOR_EVERY = display_for_every if display_for_every != 1 else optim_options["maxiter"] // 5 if optim_options["maxiter"]>5 else 1

    if exclude is not None:
        mll_params = [
            t for p_name, t in mll.named_parameters() if p_name not in exclude
        ]
    else:
        mll_params = list(mll.parameters())
    
    optimizer = None
    if custom_optimizer is None: # default state for self.optimizer in surrogate.py
        optimizer = optimizer_cls(
            params=[{"params": mll_params}],
            **_filter_kwargs(optimizer_cls, **optim_options),
        )
    else:
        optimizer = custom_optimizer
    
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         state[k] = v.to(device)
    
    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(mll, "named_parameters_and_constraints"):
        for param_name, _, constraint in mll.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    iterations = []
    t1 = time.time()

    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in mll.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **optim_options)
    )
    
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets           
    while not stop:
        optimizer.zero_grad()
        with gpt_settings.fast_computations(log_prob=approx_mll):
            output = mll.model(*train_inputs)
            # we sum here to support batch mode
            args = [output, train_targets] + _get_extra_mll_args(mll)
            loss = -mll(*args).sum()
            loss.backward()
        loss_trajectory.append(loss.item())
        for name, param in mll.named_parameters():
            param_trajectory[name].append(param.detach().clone())
        if optim_options["disp"]: # and ((i + 1) % (optim_options['maxiter']//DISPLAY_FOR_EVERY) == 0):# or i == (optim_options["maxiter"] - 1)):
            print(f"\tTrain iter: {i+1:>3}/{optim_options['maxiter']} / Loss: {loss.item():>4.3f}")
        if track_iterations:
            iterations.append(OptimizationIteration(i, loss.item(), time.time() - t1))
        optimizer.step()
        # project onto bounds:
        if bounds_:
            for pname, param in mll.named_parameters():
                if pname in bounds_:
                    param.data = param.data.clamp(*bounds_[pname])
        i += 1
        stop = stopping_criterion.evaluate(fvals=loss.detach())
    info_dict = {
        "fopt": loss_trajectory[-1],
        "wall_time": time.time() - t1,
        "iterations": iterations,
    }
    # if not optim_options['disp']:
    #     print(f"\tTrain iter: {i:>3}/{optim_options['maxiter']} / Loss: {loss.item():>4.3f}")
    return mll, info_dict, optimizer

def fit_gpytorch_scipy(
    mll: MarginalLogLikelihood,
    bounds: Optional[ParameterBounds] = None,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
    approx_mll: bool = False,
    scipy_objective: TScipyObjective = _scipy_objective_and_grad,
    module_to_array_func: TModToArray = module_to_array,
    module_from_array_func: TArrayToMod = set_params_with_array,
):# -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    r"""Fit a gpytorch model by maximizing MLL with a scipy optimizer.

    The model and likelihood in mll must already be in train mode.
    This method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A dictionary mapping parameter names to tuples of lower and upper
            bounds.
        method: Solver type, passed along to scipy.minimize.
        options: Dictionary of solver options, passed along to scipy.minimize.
        track_iterations: Track the function values and wall time for each
            iteration.
        approx_mll: If True, use gpytorch's approximate MLL computation. This is
            disabled by default since the stochasticity is an issue for
            determistic optimizers). Enabling this is only recommended when
            working with large training data sets (n>2000).

    Returns:
        2-element tuple containing
        - MarginalLogLikelihood with parameters optimized in-place.
        - Dictionary with the following key/values:
        "fopt": Best mll value.
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationIteration objects with information on each
        iteration. If track_iterations is False, will be empty.
        "OptimizeResult": The result returned by `scipy.optim.minimize`.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_scipy(mll)
        >>> mll.eval()
    """
    options = options or {}
    x0, property_dict, bounds = module_to_array_func(
        module=mll, bounds=bounds, exclude=options.pop("exclude", None)
    )
    x0 = x0.astype(np.float64)
    if bounds is not None:
        bounds = Bounds(lb=bounds[0], ub=bounds[1], keep_feasible=True)

    xs = []
    ts = []
    t1 = time.time()

    def store_iteration(xk):
        xs.append(xk.copy())
        ts.append(time.time() - t1)

    cb = store_iteration if track_iterations else None

    with gpt_settings.fast_computations(log_prob=approx_mll):
        res = minimize(
            scipy_objective,
            x0,
            args=(mll, property_dict),
            bounds=bounds,
            method=method,
            jac=True,
            options=options,
            callback=cb,
        )
        iterations = []
        if track_iterations:
            for i, xk in enumerate(xs):
                obj, _ = scipy_objective(x=xk, mll=mll, property_dict=property_dict)
                iterations.append(OptimizationIteration(i, obj, ts[i]))
    # Construct info dict
    info_dict = {
        "fopt": float(res.fun),
        "wall_time": time.time() - t1,
        "iterations": iterations,
        "OptimizeResult": res,
    }
    if not res.success:
        try:
            # Some res.message are bytes
            msg = res.message.decode("ascii")
        except AttributeError:
            # Others are str
            msg = res.message
        warnings.warn(
            f"Fitting failed with the optimizer reporting '{msg}'", OptimizationWarning
        )
    # Set to optimum
    mll = module_from_array_func(mll, res.x, property_dict)
    return mll, info_dict
