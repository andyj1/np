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
import matplotlib.pyplot as plt

from surrogate import SurrogateModel
from acquisition import AcquisitionFunction
from bo_utils import objective, contourplot
from dataset import toydataGenerator, _toydataPOST


if __name__ == '__main__':
    # toy data generation
    # load Pre-AOI and Post-AOI data
    with open('config.yml', 'r')  as file:
        cfg = yaml.load(file, yaml.FullLoader)
    x_pre, y_pre, x_post, y_post = toydataGenerator(cfg)
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # simpleplot(x_pre, y_pre, ax1, title='PRE', legend='PRE')
    # ax2 = fig.add_subplot(122)
    # simpleplot(x_post, y_post, ax2, title='POST', legend='POST')
    # fig.tight_layout()
    # plt.show()
    # contourplot(x_pre, y_pre, 'pre')
    # contourplot(x_post, y_post, 'post')    
    
    # objective: to minimize the distance from the POST to origin toward zero
    euclidean_dist = [objective(x1, x2) for x1, x2 in zip(x_post, y_post)]
    
    input_tensor = torch.cat([x_pre, y_pre], dim=1) # (N,2) dim
    target_tensor = torch.FloatTensor(euclidean_dist).unsqueeze(1) # (N,1) dim
    
    # initialize and fit model
    surrogate = SurrogateModel()
    surrogate.fit(input_tensor, target_tensor)
    # NP
    
    print(input_tensor.shape, target_tensor.shape)
    surrogate.fitNP(input_tensor, target_tensor, cfg)
    
    # loop training the model with online data
    NUM_ITER = 100
    candidates_pre = []
    candidates_post = []
    for _ in trange(NUM_ITER):
        surrogate.model.eval()
        
        # optimize acquisition function -> candidate x, y
        acq_fcn = AcquisitionFunction(model=surrogate.model, beta=0.1)
        candidate, acq_value = acq_fcn.optimize()
        
        # actual values from the objective, compute the distance
        x_new_post, y_new_post =_toydataPOST(candidate[0][0], candidate[0][1], cfg)
        
        # append to current list
        input_tensor = torch.cat([input_tensor, torch.FloatTensor(candidate)], dim=0)
        new_euc_dist_actual = torch.FloatTensor([objective(candidate[0][0], candidate[0][1])]).unsqueeze(1)
        target_tensor = torch.cat([target_tensor, new_euc_dist_actual])
        
        print('[updated] input shape:',input_tensor.shape, \
                'output tensor:',target_tensor.shape)
        # train
        surrogate.fit(input_tensor, target_tensor)
                
        # eval
        x = input_tensor
        posterior = surrogate.eval(x)
        print('posterior(',len(posterior),'):', posterior)
        