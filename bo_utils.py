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


# def plot(X, y, fig):
#     # scatter plot of inputs and real objective function
#     plt.scatter(X, y)
#     # line plot of surrogate function across domain
#     Xsamples = asarray(arange(0, 1, 0.001))
#     Xsamples = Xsamples.reshape(len(Xsamples), 1)
#     ysamples, _ = surrogate(model, Xsamples)
#     plt.plot(Xsamples, ysamples)
#     # show the plot
#     plt.show()

def simpleplot(X, y, ax, title='points', xlabel='x', ylabel='y', legend='', label=''):
    indices_to_order_by = X.squeeze().argsort() # dim squeeze: 10x1 -> 10
    x_ordered = X[indices_to_order_by]
    y_ordered = y[indices_to_order_by]
    ax.plot(x_ordered, y_ordered, marker="o", markerfacecolor="r", label=label)
    # ax.scatter(X, y, label=legend)
    ax.legend(loc='best')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
def contourplot(x, y, label):
    xx, yy = np.meshgrid(x, y)
    zz = np.sqrt(xx**2+yy**2) #np.linalg.norm((xx,yy))
    fig, ax = plt.subplots(1,1)
    cp = ax.contourf(xx, yy, zz, label=label)
    fig.colorbar(cp)
    ax.set_title('contour plot - '+label)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    plt.show()