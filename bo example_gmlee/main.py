import numpy as np
import matplotlib.pyplot as plt

from utils import generatePoints2D
from randomfunctions import simple

from gpytorch.mlls import ExactMarginalLogLikelihood

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

if __name__ == '__main__':
    inputs = generatePoints2D(100, mu = 10, sigma = 30)
    observations_x = simple(inputs[0],dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 30, hv = True)
    observations_y = simple(inputs[1],dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 30, hv = False)
    print()

    in_dist = np.asarray([])
    for x, y in zip(observations_x, observations_y) :
        in_dist = np.append(in_dist, np.linalg.norm((x,y)))

    out_dist = np.asarray([])
    for x, y in zip(observations_x, observations_y) :
        out_dist = np.append(out_dist, -np.linalg.norm((x,y)))
    print('out_dist', np.sum(out_dist))

    fig, ax = plt.subplots()
    # ax.scatter(inputs[0], out_dist, label = 'x-distance')

    ax.scatter(inputs[0], inputs[1], label = 'inputs')
    ax.scatter(observations_x, observations_y, label ='outputs')


    # model = GaussianProcessRegressor()
    # model.fit(np.concatenate(inputs, 1), out_dist)
    inputs = np.concatenate(inputs, 1)
    candidates_x = []
    candidates_y = []
    cand_outs_x = []
    cand_outs_y = []
    for _ in range(200):
        out_dist = out_dist.reshape(-1, 1)
        gp = SingleTaskGP(torch.tensor(inputs), torch.tensor(out_dist))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        UCB = UpperConfidenceBound(gp, beta=0.5) # analytic.py in source code has 'maximize'=True, so objective is '-distance'

        bounds = torch.stack([torch.ones(2) * -100, torch.ones(2) * 100])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        inputs = np.concatenate((inputs, candidate), 0)
        candidate = candidate.reshape(-1,1)
        cand_out_x = simple(candidate[0].reshape(-1,1),dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 10)
        cand_out_y = simple(candidate[0].reshape(-1,1),dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 10)
        out_dist = np.append(out_dist, -np.linalg.norm((cand_out_x, cand_out_y)))
        cand_outs_x.append(cand_out_x)
        cand_outs_y.append(cand_out_y)

        candidates_x.append(candidate[0].numpy())
        candidates_y.append(candidate[1].numpy())
        print(candidate, out_dist[-1])

    ax.scatter(candidates_x, candidates_y, label ='candidates')
    ax.scatter(cand_outs_x, candidates_y, label ='candidates_output')
    ax.legend(fontsize=12, loc='upper left')  # legend position
    plt.ylim([-120, 120])
    plt.xlim([-120, 120])
    plt.savefig('test.png')
