''' Procedure
1. data
    a. X
        - load PRE (assume Gaussian)
        - load post: assume some reflow oven effect (angular and distance shift) from PRE
    b. Y
        - compute the Euclidean distance to use as the target_tensor 
2. load model
3. train model with X and Y
4. sample a new PRE value from the optimized acquisition
5. determine corresponding POST values 
6. compute the Euclidean distance to use and append as the new target_tensor value
7. visualize altogether iteratively
'''

import math, os, sys

import yaml
from tqdm import trange
import torch
import numpy as np

from surrogate import SurrogateModel
from acquisition import AcquisitionFunction
import bo_utils as boutils # objective
import np_utils as nputils #
import viz_utils as vizutils # contourplot, draw_graphs
from dataset import toydataGenerator, reflow_oven


# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

if __name__ == '__main__':
    """ load Pre-AOI and Post-AOI data """
    with open('config.yml', 'r')  as file:
        cfg = yaml.load(file, yaml.FullLoader)
    x_pre, y_pre, x_post, y_post = toydataGenerator(cfg)
    print('[INFO] pre distance:', np.sqrt(np.sum(x_pre.numpy()**2 + y_pre.numpy()**2)),
          'post distance:', np.sqrt(np.sum(x_post.numpy()**2 + y_post.numpy()**2)))
    
    """ plot data """
    _bin = np.linspace(-120,120,60)
    _bins_x, _bins_y = np.meshgrid(_bin, _bin)
    _bins = np.concatenate([_bins_x.reshape(-1,1), _bins_y.reshape(-1,1)],1)
    vizutils.contourplot(_bin, _bin, 'Train data', x_pre = x_pre, y_pre = y_pre, x_post = x_post, y_post = y_post, cfg = cfg)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # simpleplot(x_pre, y_pre, ax1, title='PRE', legend='PRE')
    # ax2 = fig.add_subplot(122)
    # simpleplot(x_post, y_post, ax2, title='POST', legend='POST')
    # fig.tight_layout()
    # plt.show()
    # contourplot(x_pre, y_pre, 'pre')
    # contourplot(x_post, y_post, 'post')    
    
    # objective function
    # - to minimize the distance from the POST to origin toward zero
    euclidean_dist = [boutils.objective(x1, x2) for x1, x2 in zip(x_post, y_post)]
    
    # make into proper dimension for SurrogateModel (e.g. SingleTaskGP)
    input_tensor = torch.cat([x_pre, y_pre], dim=1) # (N,2) dim
    target_tensor = torch.FloatTensor(euclidean_dist).unsqueeze(1) # (N,1) dim
    # print(input_tensor.shape, target_tensor.shape)
    
    # initialize and fit model
    surrogate = SurrogateModel(cfg=cfg, neural=True)
    # surrogate.fitGP(input_tensor, target_tensor)
    # for NP model, set neural to True
    # surrogate.model.train() # unncessary for GPyTorch(BoTorch)-based model
    surrogate.fitNP(input_tensor, target_tensor, cfg)
        
    # training loop
    NUM_ITER = cfg['num_iter']
    candidates_pre, candidates_post = [], []
    for iter in trange(NUM_ITER):
        # surrogate.model.eval() # unncessary for GPyTorch(BoTorch)-based model
        
        # optimize acquisition function -> candidate x, y
        acq_fcn = AcquisitionFunction(model=surrogate.model, beta=0.1)
        candidate, acq_value = acq_fcn.optimize()
        
        # actual values from the objective, compute the distance
        x_new_post, y_new_post = reflow_oven(candidate[0][0], candidate[0][1], cfg, toy=True)
        
        # append to current input and target tensors
        new_input = torch.FloatTensor(candidate)
        new_target = torch.FloatTensor([boutils.objective(candidate[0][0], candidate[0][1])]).unsqueeze(1)
        input_tensor = torch.cat([input_tensor, new_input], dim=0)
        target_tensor = torch.cat([target_tensor, new_target], dim=0)
        print('[INFO] new input shape:',input_tensor.shape, 'newoutput tensor:',target_tensor.shape)
        
        # re-train
        surrogate.fit(input_tensor, target_tensor)
        
        # eval
        # x = input_tensor
        # posterior = surrogate.eval(x)
        # print('posterior(',len(posterior),'):', posterior)
        
        # for visualization
        # if iter % 5 == 0 and iter > 0:
        #     vizutils.draw_graphs(input_tensor[:cfg['num_samples']], input_tensor[cfg['num_samples']:],
        #                 x_post, y_post, cfg, iter)
        #     _dim = int(np.sqrt(_bins.shape[0]))
        #     vizutils.contourplot(_bin, _bin, 'Results_%d'%iter,
        #                 x_pre = input_tensor[:,0], y_pre = input_tensor[:,1], x_post = x_post, y_post = y_post,
        #                 cfg = cfg)
    
    posterior, bnds = surrogate.eval(torch.tensor(_bins).to(torch.float32))
    print('posterior(',len(posterior),'):', min(bnds[1] - bnds[0]), max(bnds[1] - bnds[0]))
    euclidean_torch = lambda X :torch.sqrt(torch.mean(torch.sum(X**2,1),0))
    _dim = int(np.sqrt(_bins.shape[0]))
    vizutils.contourplot(_bin, _bin, 'Results_final', x_pre = input_tensor[:,0], y_pre = input_tensor[:,1], x_post = x_post, y_post = y_post,
                # bnds = (bnds[1] - bnds[0]).reshape(_dim,_dim),
                cfg = cfg)

    train_data, BO_data = input_tensor[:cfg['num_samples']], input_tensor[cfg['num_samples']:]
    print("Train data mean, std : ", torch.mean(train_data, 0), torch.std(train_data, 0))
    print("Train data distance, loss : ", euclidean_torch(train_data), torch.mean(target_tensor[:cfg['num_samples']]))
    print("BO data mean, std : ", torch.mean(BO_data, 0), torch.std(BO_data, 0))
    print("BO data distance, loss : ", euclidean_torch(BO_data), torch.mean(target_tensor[cfg['num_samples']:]))
    print('\n[INFO] Finished.')
        