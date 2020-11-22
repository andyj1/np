from numpy.random import normal, random
from scipy.stats import norm
from warnings import catch_warnings, simplefilter
import matplotlib.pyplot as plt
from math import pi, sin
from numpy import arange, argmax, asarray, vstack
import numpy as np

# currently only takes into account positions in the x and y direction
def objective(x, y):
    # noise=0.1
    # noise = normal(loc=0, scale=noise)
    # return (x**2 * sin(5 * pi * x)**6.0) + noise
    return np.linalg.norm((x,y))    

'''
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)  # for GPR
        # forward pass returns mean and std dev for NPR
        # return model(X, return_std=True)


def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs


def opt_acquisition(X, model, minval, maxval):
    # random search, generate random samples
    # Xsamples = random(100)
    Xsamples = arange(minval, maxval, 1e-1)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix, 0]
'''
