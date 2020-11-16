
# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
import numpy as np
from numpy.random import normal,random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
import matplotlib.pyplot as plt

import os
result_dir = 'gpr_example_results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir, mode=0o755)
filename = 0

# objective function
def objective(x, noise=0.1):
	noise = normal(loc=0, scale=noise)
	return (x**2 * sin(5 * pi * x)**6.0) + noise

# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)

# probability of improvement acquisition function
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

# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
	Xsamples = random(100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = np.argmax(scores)
	return Xsamples[ix, 0]

# plot real observations vs surrogate function
def plot(X, y, model, label):
	print('plotting...')
	global filename, result_dir
	# scatter plot of inputs and real objective function
	plt.scatter(X, y, marker='o', color='k', label="all")
	# line plot of surrogate function across domain
	Xsamples = np.asarray(np.arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	plt.plot(Xsamples, ysamples, label="surrogate samples_"+label)
	plt.legend(loc='best')
	plt.savefig(f'{result_dir}/{filename}.png')
	filename += 1
	# show the plot
	# plt.show()

if __name__=='__main__':
    # sample the domain sparsely with noise
    X = random(100)
    y = np.asarray([objective(x) for x in X])
    # reshape into rows and cols
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)
    # define the model
    model = GaussianProcessRegressor()
    # fit the model
    model.fit(X, y)
    # plot before hand
    plot(X, y, model, label='before')
    # perform the optimization process
    for i in range(100):
        # select the next point to sample
        x = opt_acquisition(X, y, model)
        # sample the point
        actual = objective(x)
        # summarize the finding
        est, var = surrogate(model, [[x]])
        print('x: %.3f, surrogate[mu(var)]: %.3f (%.3f), actual(obj): %.3f' % (x, est, var, actual))
        # add the data to the dataset
        X = np.vstack((X, [[x]]))
        y = np.vstack((y, [[actual]]))
        # update the model
        model.fit(X, y)

    # plot all samples and the final surrogate function
    plot(X, y, model, label='after')
    # best result
    ix = np.argmax(y)
    print('Best Result: x: %.3f, y: %.3f' % (X[ix], y[ix]))