import math
import os
import sys
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import trange

import bo_utils
import np_utils
import viz_utils  # contourplot, draw_graphs
from acquisition import AcquisitionFunction
from dataset import getMOM4data, getTOYdata, reflow_oven
from surrogate import SurrogateModel

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float
obj = bo_utils.objective


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', '--TOY', action='store_true', help='sets to toy dataset')
    parser.add_argument('--load', help='destination to model dict')
    args = parser.parse_args()
    
    print('='*10,'test','='*10)

    with open('config.yml', 'r')  as file:
        cfg = yaml.load(file, yaml.FullLoader)
    if args.toy:
        print('Loading Toy data...')
        x_pre, y_pre, x_post, y_post = getTOYdata(cfg)
    else:
        print('Loading MOM4 data...')
        x_pre, y_pre, x_post, y_post = getMOM4data(cfg)
    NUM_EPOCH = cfg['train']['num_epoch']
    
    ''' objective function: to minimize the distance from the POST to origin toward zero '''
    euclidean_dist = [-obj(x1, x2) for x1, x2 in zip(x_post, y_post)]
    
    ''' make into proper dimension for SurrogateModel (e.g., SingleTaskGP) '''
    input_tensor = torch.cat([x_pre, y_pre], dim=1)                 # (N,2) dim
    target_tensor = torch.FloatTensor(euclidean_dist).unsqueeze(1)  # (N,1) dim

    # initialize model and likelihood
    surrogate = SurrogateModel(epochs=NUM_EPOCH)
    EPOCH_FROM = -1
    if not args.load.empty():
        checkpoint = torch.load(args.load)
        surrogate.model.load_state_dict(checkpoint['model'])
        surrogate.optimizer.load_state_dict(checkpoint['optimizer'])
        EPOCH_FROM = int(args.load.split('.pt')[-1])
    start = time.time()
    surrogate.fitGP(input_tensor, target_tensor, epoch=0)
    print(f'[INFO] initial train time: {time.time()-start:.3f} sec')
    
    # training loop
    fig = plt.figure()
    ax = fig.add_subplot()
    candidates_pre, candidates_post = [], []
    t = trange(NUM_EPOCH)
    for epoch in t:

        # optimize acquisition functions and get new observations
        acq_fcn = AcquisitionFunction(cfg=cfg, model=surrogate.model)
        start = time.time()
        candidate, acq_value = acq_fcn.optimize()
                
        # actual values from the objective, compute the distance
        x_new_pre, y_new_pre = candidate[0] # unravel tensor to numpy floats
        x_new_post, y_new_post = reflow_oven(x_new_pre, y_new_pre)

        pre_dist = obj(x_new_pre, y_new_pre)
        post_dist = obj(x_new_post, y_new_post)
        # append distance measures
        candidates_pre.append(pre_dist)
        candidates_post.append(post_dist)
        
        # update input and target tensors
        input_tensor = torch.cat([input_tensor, candidate], dim=0)
        # since mll is maximized, purposely negate objective value
        new_target = torch.from_numpy(np.array([-post_dist])).unsqueeze(1)
        target_tensor = torch.cat([target_tensor, new_target], dim=0)
        
        t.set_description(desc=f'[INFO] Epoch {epoch+1} / train time: {time.time()-start:.3f} sec, pre_dist: {pre_dist:.3f}, post_dist: {post_dist:.3f}', refresh=False)

        # re-initialize the models so they are ready for fitting on next iteration
        # and re-train
        surrogate.fitGP(input_tensor, target_tensor, epoch=epoch)
        
        # eval
        # x = input_tensor
        # posterior = surrogate.eval(x)
        # print('posterior(',len(posterior),'):', posterior)
        
        # for visualization
        # if iter % 5 == 0 and iter > 0:
        #     viz_utils.draw_graphs(input_tensor[:cfg['num_samples']], input_tensor[cfg['num_samples']:],
        #                 x_post, y_post, cfg, iter)
        #     _dim = int(np.sqrt(_bins.shape[0]))
        #     viz_utils.contourplot(_bin, _bin, 'Results_%d'%iter,
        #                 x_pre = input_tensor[:,0], y_pre = input_tensor[:,1], x_post = x_post, y_post = y_post,
        #                 cfg = cfg)

        
        ax.scatter(x_new_pre, y_new_pre, s=10, alpha=(epoch+1)*1/NUM_EPOCH, color='r')
        ax.scatter(x_new_post, y_new_post, s=10, alpha=(epoch+1)*1/NUM_EPOCH, color='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Pre -> Post')
    fig.savefig('img.png')
    
    for pre, post in zip(candidates_pre, candidates_post):
        print(f'Distance: {pre:.3f} -> {post:.3f}')
    # posterior, bnds = surrogate.eval(torch.tensor(_bins).to(torch.float32))
    # print('posterior(',len(posterior),'):', min(bnds[1] - bnds[0]), max(bnds[1] - bnds[0]))
    # euclidean_torch = lambda X :torch.sqrt(torch.mean(torch.sum(X**2,1),0))
    # _dim = int(np.sqrt(_bins.shape[0]))
    # viz_utils.contourplot(_bin, _bin, 'Results_final', x_pre = input_tensor[:,0], y_pre = input_tensor[:,1], x_post = x_post, y_post = y_post,
    #             # bnds = (bnds[1] - bnds[0]).reshape(_dim,_dim),
    #             cfg = cfg)

    # train_data, BO_data = input_tensor[cfg['train']['num_samples']], input_tensor[cfg['train']['num_samples']:]
    # print("Train data mean, std : ", torch.mean(train_data, 0), torch.std(train_data, 0))
    # print("Train data distance, loss : ", euclidean_torch(train_data), torch.mean(target_tensor[:cfg['num_samples']]))
    # print("BO data mean, std : ", torch.mean(BO_data, 0), torch.std(BO_data, 0))
    # print("BO data distance, loss : ", euclidean_torch(BO_data), torch.mean(target_tensor[cfg['num_samples']:]))
    # print('\n[INFO] Finished.')
        