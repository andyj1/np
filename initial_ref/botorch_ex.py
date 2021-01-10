import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

# Fit a model:
train_X = torch.rand(10, 2)
Y = 1 - torch.norm(train_X - 0.5, dim=-1) + 0.1 * torch.rand(10)
train_Y = (Y - Y.mean()) / Y.std()
train_Y = train_Y.unsqueeze(1)
print('example shapes:',train_X.shape, train_Y.shape)

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# Construct an acquisition function:
from botorch.acquisition import UpperConfidenceBound
UCB = UpperConfidenceBound(gp, beta=0.1)

# Optimize the acquisition function:
from botorch.optim import optimize_acqf
bounds = torch.stack([torch.zeros(2), torch.ones(2)])
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
print(candidate)

'''
def fit_gpytorch_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    r"""Fit hyperparameters of a gpytorch model.

    Optimizer functions are in botorch.optim.fit.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        optimizer: The optimizer function.
        kwargs: Arguments passed along to the optimizer function.

    Returns:
        MarginalLogLikelihood with optimized parameters.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> fit_gpytorch_model(mll)
    """
    mll.train()
    mll, _ = optimizer(mll, track_iterations=False, **kwargs)
    mll.eval()
    return mll
'''
