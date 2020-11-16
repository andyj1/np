''' Pipeline 
1. retrieve data from CSV -> X, y 
    * Y should ideally be from an objective function f that maps X to y (f: X -> y)
    ex) X = PRE_R
        y = POST_R
2. define a surrogate model
    ex) GPR
        NPR
3. train the surrogate model with context data (X, y) pairs
4. from the optimized acquisition function (sampling from the optimized/trained model),
    sample for the next input X_new
    * inputs are random search (for now, but will be within the defined range for optimization)
    * also takes in the trained model
    * returns X_new that yields the best probability within the defined bound
5. from the objective function, retrieve the sample point for the newly searching X, (X_new)
    * objective is to minimize the offset distance, toward zero (R=0)
6. predict mean and std dev with the trained surrogate model for that new sample X_new
7. compare Y from the objective(X_new) and estimated mean from the surrogate model
8. update surrogate model with the new sample and objective(X_new) output
'''

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import arange, argmax, asarray, vstack
from numpy.random import normal, random
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.gaussian_process import GaussianProcessRegressor  # surrogate model class

from bo_utils import acquisition, opt_acquisition, plot, surrogate, objective
from dataset import getMOM4data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.set_option('display.max_columns', None)

file = 'MOM4_data.csv'
getMOM4data(file=file)
# pseudo-objective function for X and y
obj_dict = dict()  # key: str(x), value: y
for (x, y) in list(zip(X, Y)):
    print('x: %f, y: %f' % (x, y))
    try:
        obj_dict[str(x)] = y
    except KeyError:
        obj_dict[str(x)] = 0.

# define the model (surrogate)
model = GaussianProcessRegressor()
# model = NeuralProcessRegressor()

# train the model (GaussianProcessRegressor)
model.fit(X, Y)

# train the model (NeuralProcessRegressor)
# model.train(epochs, n_context, all_x_np, all_y_np)
# vals = np.arange(min(all_x_np), max(all_x_np), 1e-3)
# x_grid = torch.from_numpy(vals.reshape(-1, 1).astype(np.float32))

# hidden_dim, decoder_dim, z_samples = 10, 15, 20  # 10, 15, 20
# model = NP(hidden_dim, decoder_dim, z_samples).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# n_context = np.random.randint(20, 30) # a random number between the two numbers
# n_context = 20
# train(10**5, n_context)


# def objective(x): return obj_dict[str(x)]


# plot all
plot(X, Y, model)
# perform the optimization process
# for i in range(100):
for i in arange(min(X), max(X), 1e-1):
    # select the next point to sample
    x = opt_acquisition(X, model, min(X), max(X))
    # sample the point
    actual = objective(x)
    # summarize the finding
    est, _ = surrogate(model, [[x]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
    # add the data to the dataset
    X = vstack((X, [[x]]))
    Y = vstack((Y, [[actual]]))
    # update the model
    model.fit(X, Y)

# plot all samples and the final surrogate function
plot(X, Y, model)
# best result
ix = argmax(Y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], Y[ix]))
