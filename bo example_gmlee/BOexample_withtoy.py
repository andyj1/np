import numpy as np
import matplotlib.pyplot as plt
import time

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

train_sample = 100
test_iter = 50


def selfalignment(pre_chip, device):
	x_offset = torch.FloatTensor([50])
	y_offset = torch.FloatTensor([50])
	post_chip = pre_chip + torch.tile(torch.FloatTensor([x_offset, y_offset]).to(device), (pre_chip.shape[0], 1))
	return post_chip

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# generate inputs	
	l = torch.normal(mean=0, std=10, size=(train_sample, 1))
	w = torch.normal(mean=0, std=10, size=(train_sample, 1))
	inputs = torch.cat([l, w], dim=1).to(device)

	# self alignment
	obs = selfalignment(inputs, device)

	# compute distances
	in_dist = torch.FloatTensor([])
	out_dist = torch.FloatTensor([])
	
	for x, y in inputs:
		in_dist = torch.cat([in_dist, torch.FloatTensor([np.linalg.norm((x.cpu(), y.cpu()))])])

	for x, y in obs:
		out_dist = torch.cat([out_dist, torch.FloatTensor([-np.linalg.norm((x.cpu(),y.cpu()))])])
	in_dist = in_dist.unsqueeze(1).to(device)
	out_dist = out_dist.unsqueeze(1).to(device)
	
	# import sys; sys.exit(1)
  
	fig = plt.figure()
	ax = fig.add_subplot()
	for t_iter in range(test_iter):
		# GP model
		print(inputs.shape, out_dist.shape)
		gp = SingleTaskGP(inputs, out_dist)
		mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
		fit_gpytorch_model(mll)
  
  		# acquision
		UCB = UpperConfidenceBound(gp, beta=0.5)
		bounds = torch.stack([torch.ones(1) * -20, torch.ones(1) * 20]).to(device)
		candidate, acq_value = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
		candidate_obs = selfalignment(candidate, device)
		out_dist = np.append(out_dist, -np.linalg.norm((candidate_obs[0,0].cpu(), candidate_obs[0,1].cpu())))

		print(candidate, candidate_obs)
		# append
		inputs = torch.cat([inputs, candidate], dim=0)
		obs = torch.cat([obs, candidate_obs], dim=0)
        
		ax.scatter(candidate[0], candidate[1], label ='candidates')
		ax.scatter(candidate_obs[0], candidate_obs[1], label ='candidates_output')
		ax.legend(fontsize=12, loc='upper left')  # legend position

	fig, ax = plt.subplots()
	ax.scatter(inputs[:train_sample,0], inputs[:train_sample,1], label = 'PRE')
	ax.scatter(obs[:,0], obs[:,1], label ='POST')
	ax.legend(fontsize=12, loc='upper left')  # legend position
	plt.ylim([-120, 120])
	plt.xlim([-120, 120])
	plt.savefig('fig_last.png', dpi=300)
