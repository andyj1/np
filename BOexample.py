import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import sys
# toy data for X, y
def generatePoints2D(num, mu = 0, sigma = 20) :
	x = np.random.normal(mu, sigma, num).reshape(-1,1)
	y = np.random.normal(mu, sigma, num).reshape(-1,1)
	return x, y

# appends offset in the pre-aoi stage
def simple(points, dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 30, i = True) :
	num = points.shape[0]
	r_dist = np.random.normal(dist_mu, dist_sigma, num)
	r_degree = np.random.normal(degree_mu, degree_sigma, num)
	if i :
		return points[:,0] + r_dist * np.cos(r_degree * np.pi / 180)
	else :
		return points[:,0] + r_dist * np.sin(r_degree * np.pi / 180)



if __name__ == '__main__':
	inputs = generatePoints2D(100, mu = 10, sigma = 30)
	observations_x = simple(inputs[0],dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 30, i = True)
	observations_y = simple(inputs[1],dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 30, i = False)

	in_dist = np.asarray([])
	for x, y in zip(observations_x, observations_y) :
		in_dist = np.append(in_dist, np.linalg.norm((x,y)))
	print('in_dist', in_dist[0:5])
	out_dist = np.asarray([])
	for x, y in zip(observations_x, observations_y) :
		out_dist = np.append(out_dist, -np.linalg.norm((x,y)))
	# print('out_dist', out_dist[0:5])
	print('out_dist', np.sum(out_dist))

	fig, ax = plt.subplots()

	ax.scatter(inputs[0], inputs[1], label = 'inputs')
	ax.scatter(observations_x, observations_y, label ='outputs')

	inputs = np.concatenate(inputs, 1)
	candidates_x = []
	candidates_y = []
	cand_outs_x = []
	cand_outs_y = []
	for _ in range(100):
		out_dist = out_dist.reshape(-1, 1)
		print('input tensor:',inputs.shape,' out_dist tensor:', out_dist.shape)

		gp = SingleTaskGP(torch.tensor(inputs), torch.tensor(out_dist))
		mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
		fit_gpytorch_model(mll)

		UCB = UpperConfidenceBound(gp, beta=0.5)

		bounds = torch.stack([torch.ones(2) * -100, torch.ones(2) * 100])
		candidate, acq_value = optimize_acqf(
			UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
		)

		inputs = np.concatenate((inputs, candidate), 0)
		candidate = candidate.reshape(-1,1)

		# append PRE to the list
		candidates_x.append(candidate[0].numpy())
		candidates_y.append(candidate[1].numpy())

		# get and append POST
		cand_out_x = simple(candidate[0].reshape(-1,1),dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 10)
		cand_out_y = simple(candidate[0].reshape(-1,1),dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 10)
		cand_outs_x.append(cand_out_x)
		cand_outs_y.append(cand_out_y)
		
		# append distance (objective)
		out_dist = np.append(out_dist, -np.linalg.norm((cand_out_x, cand_out_y)))
		
		print('candidate:',candidate, 'out_dist', out_dist[-1])
	
	# PRE
	ax.scatter(candidates_x, candidates_y, label ='candidates')
	# POST
	ax.scatter(cand_outs_x, candidates_y, label ='candidates_output')
	ax.legend(fontsize=12, loc='upper left')  # legend position
	plt.ylim([-120, 120])
	plt.xlim([-120, 120])
	plt.show()
	