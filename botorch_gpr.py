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
	cfg, args = parse_utils.parse()
 
	device = torch.device(cfg['train_cfg']['device'])
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
	x_bound_min, x_bound_max = -10, 10
	y_bound_min, y_bound_max = -10, 10
	pbounds = [[x_bound_min, x_bound_max], [y_bound_min, y_bound_max]]

	fig, ax = plt.subplots()
	save_image_path = './fig_botorch'
	os.makedirs(save_image_path, exist_ok=True, mode=0o755)
 
	# print('inputs:',inputs.shape) # [100, 2]
	# import sys; sys.exit(1)
 
	ax = plot_grid(ax, inputs[:, 0], inputs[:, 1], pbounds, num_dim, 'GP', iteration=0)
	plt.pause(0.00001)
 
	inputs = inputs.to(device)
	outputs = outputs.to(device)
	candidates = []
	avg_time = 0
	for t_iter in range(1, test_iter+1, 1):
		plt.clf()
  
  		# ====================================
		start = time.time()
		gp = SingleTaskGP(inputs, outputs).to(device)
		mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
		mll.to(inputs.device)
		fit_gpytorch_model(mll)

		UCB = UpperConfidenceBound(gp, beta=beta, maximize=False).to(device)

		# dim = inputs.shape[1]
		# bounds = torch.stack([torch.ones(dim) * -10, 
        #                 		torch.ones(dim) * 10])
		
		min_bounds = torch.cat([torch.ones(1) * x_bound_min, torch.ones(1) * y_bound_min])
		max_bounds = torch.cat([torch.ones(1) * x_bound_max, torch.ones(1) * y_bound_max]) 
		bounds = torch.stack([min_bounds, max_bounds]).to(device)
  
		candidate, acq_value = optimize_acqf(
			UCB, bounds=bounds, q=1, num_restarts=acq_cfg['num_restarts'], raw_samples=acq_cfg['raw_samples'],
		)

		end = time.time()
		print(f'iter {t_iter} | data size: {inputs.shape[0]}, elapsed time: {(end-start):.4f} sec')
		# ====================================
  
		candidate = candidate.to(device)		
		inputs = torch.vstack((inputs, candidate))
		inputs_shifted, method = self_alignment.self_alignment(inputs)
		outputs = xy_to_L2distance(inputs_shifted)

		# print(candidate)
  
		candidate = candidate.cpu().numpy()
		# plt.scatter(candidate[0][0], candidate[0][1], s=30, marker='x', c='k', label='candidate')
		
		ax = plot_grid(ax, inputs[:, 0], inputs[:, 1], pbounds, num_dim, 'GP', t_iter)

		candidates.append(candidate)
		for i, candidate in enumerate(candidates):
			if i < 10:      
				plt.scatter(candidate[0][0], candidate[0][1], s=30, marker='x', c='k', label=f'{i+1}')
			else:
				plt.scatter(candidate[0][0], candidate[0][1], s=30, marker='x', c='k', label='')
		plt.legend(loc='best')
		plt.pause(0.0001)
	
		plt.savefig(os.path.join(save_image_path, f'{str(device).split(":")[0]}_{num_dim}_{len(inputs)}.png'))
  
		avg_time += (end-start)
	avg_time /= test_iter
	print(f'avg time: {avg_time:.4f}')
 
	for i, candidate in enumerate(candidates):
		plt.scatter(candidate[0][0], candidate[0][1], s=30, marker='x', c='k', label=f'{i+1}')
	plt.pause(5)
    
  
		