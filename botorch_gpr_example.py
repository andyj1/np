import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

input_dim = 2
num_samples = 100
train_X = torch.rand(num_samples, input_dim)
Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
Y += 0.1 * torch.rand_like(Y)
train_Y = (Y - Y.mean()) / Y.std()

# single task gp
gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# acquisition
from botorch.acquisition import UpperConfidenceBound
UCB = UpperConfidenceBound(gp, beta=0.5, maximize=True) # maximize

# optimize
from botorch.optim import optimize_acqf
bounds = torch.stack([torch.zeros(input_dim), torch.ones(input_dim)])
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
# print(torch.cat((train_X, train_Y), dim=-1))
print(candidate, acq_value)

UCB = UpperConfidenceBound(gp, beta=0.5, maximize=False)    # minimize
bounds = torch.stack([torch.zeros(input_dim), torch.ones(input_dim)])
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
# print(torch.cat((train_X, train_Y), dim=-1))
print(candidate, acq_value)