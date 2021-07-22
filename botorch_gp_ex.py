import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import data as custom
import parse_utils
from toy import self_alignment
from visualize_objective_space import plot_grid

test_iter = 50

if __name__ == '__main__':
	device = torch.device('cuda')
 
	cfg, args = parse_utils.parse()
	train_cfg, data_cfg, acq_cfg = cfg['train_cfg'], cfg['data_cfg'], cfg['acq_cfg']
	num_samples = cfg['train_cfg']['num_samples']
	num_workers = cfg['train_cfg']['num_workers']
	num_dim = cfg['train_cfg']['x_dim']
	data_cfg = cfg['data_cfg']
	beta = cfg['acq_cfg']['beta']
	dataset = custom.CustomData(input_dim=num_dim, num_samples=num_samples, type='toy', cfg=data_cfg)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=4)
	sample_batch = next(iter(dataloader))
	inputs, outputs = sample_batch[0], sample_batch[1]
	print('[PRE]:', inputs.shape)
	print('[POST DISTANCE]:', outputs.shape)
 
	xy_to_L2distance = lambda inputs: torch.linalg.norm(inputs, dim=1).unsqueeze(-1)
	bound_min, bound_max = -10, 10

	fig, ax = plt.subplots()
	save_image_path = './fig_botorch'
	os.makedirs(save_image_path, exist_ok=True, mode=0o755)
 
	# if num_dim == 2:
	iteration = 0
	ax = plot_grid(fig, ax, inputs[:, 0], inputs[:, 1], bound_min, bound_max, save_image_path, iteration, num_dim)
	plt.colorbar()
	plt.pause(0.001)
	iteration += 1
 
	inputs = inputs.to(device)
	outputs = outputs.to(device)
	avg_time = 0
	for t_iter in range(test_iter):
		start = time.time()
		plt.clf()
		start_time = time.time()
		gp = SingleTaskGP(inputs, outputs).to(device)
		mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
		mll.to(inputs.device)
		fit_gpytorch_model(mll)

		UCB = UpperConfidenceBound(gp, beta=beta, maximize=False).to(device)

		dim = inputs.shape[1]
		bounds = torch.stack([torch.ones(dim) * bound_min, torch.ones(dim) * bound_max]).to(device)
		candidate, acq_value = optimize_acqf(
		UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
		)
		end_time = time.time()
		print(f'iter {t_iter+1} | data size: {inputs.shape[0]}, elapsed time: {(end_time-start_time):.4f} sec')
		candidate = candidate.to(device)
  
		inputs = torch.vstack((inputs, candidate))
		inputs_shifted, method = self_alignment.self_alignment(inputs)
		outputs = xy_to_L2distance(inputs_shifted)

		# print(candidate)
  
		candidate = candidate.cpu().numpy()
		plt.scatter(candidate[0][0], candidate[0][1], s=50, marker='x', c='k', label='candidate')
		# if num_dim == 2:
		ax = plot_grid(fig, ax, inputs[:, 0], inputs[:, 1], bound_min, bound_max, save_image_path, iteration, num_dim)
		plt.legend()
		plt.pause(0.001)
		iteration += 1
  
		end = time.time()
		avg_time += (end-start)
	avg_time /= test_iter
	print(f'avg time: {avg_time:.4f}')
  
		