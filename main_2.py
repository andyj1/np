from operator import concat
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from botorch.sampling import IIDNormalSampler
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import trange
import sys
# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

from bo_utils import simpleplot, objective
from dataset import toydataGenerator, toydataPOST
# import sys
# sys.exit()

''' data generation (toy) 
x_before, y_before: 2-D points (initially defined)
x_after, y_after: 2-D points with offset in distance and angle
''' 
with open('config.yml', 'r')  as file:
    cfg = yaml.load(file, yaml.FullLoader)
x_before, y_before, x_after, y_after = toydataGenerator(cfg) # PRE, POST toy

# objective function
# * to minimize the distance from the POST to zero
euc_dist = [objective(x, y) for x, y in zip(x_after, y_after)]

# given PRE data, the surrogate model is to minimize the objective (Euc dist)
# make pre_data and objective output both 2-D
# pre_data: (N x 2), euc_dist: (N, 1)
pre_data = torch.cat((x_before, y_before), dim=1).squeeze()
distance = torch.FloatTensor(euc_dist).unsqueeze(1)
print('initial shapes:', pre_data.shape, distance.shape)

NUM_ITER = 100
candidate_x, candidate_y = [], []
for i in trange(NUM_ITER):
    # 1. fit a Gaussian model to data 
    model = SingleTaskGP(pre_data, distance).to(device)
    model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))    
    mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    mll.to(x_before) # set mll and all submodules to the specified dtype and device
    fit_gpytorch_model(mll) # -> MarginalLogLikelihood

    # construct an acquisition function
    UCB = UpperConfidenceBound(model, beta=0.1)

    # set bounds for optimization
    d = 2 # dimension
    bounds = torch.stack([-torch.ones(d), torch.ones(d)])

    # 2. optimize the acquisition function
    candidate, acq_value = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)

    print('candidates',candidate, 'acquisition values:',acq_value)
    # store candidates from acquisitions for plotting
    # PRE
    candidate_x.append(candidate[0][0])
    candidate_y.append(candidate[0][1])
    # POST
    x_after_new, y_after_new = toydataPOST(candidate[0][0], candidate[0][1], cfg)

    pre_data = torch.cat((pre_data, candidate)).to(device)
    euc_dist_to_add = objective(x_after_new, y_after_new)
    euc_dist.append(euc_dist_to_add)
    distance = torch.FloatTensor(euc_dist).unsqueeze(1)
    print('pre_data:', pre_data.shape, 'distance:', distance.shape)

    

# visualize input points
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
simpleplot(x_before, y_before, ax, title='input', xlabel='x', ylabel='y', legend='input')
simpleplot(x_after, y_after, ax, legend='input (w/ offset')
# keep plotting newly acquired POST (X, Y)
simpleplot(candidate_x.to_, candidate_y, ax, legend='input (w/ offset')

print('candidate_x:',candidate_x)
print('candidate_y:',candidate_y)
    
    










