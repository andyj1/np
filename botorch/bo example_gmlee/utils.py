import numpy as np

def generatePoints2D(num, mu = 0, sigma = 20) :
	x = np.random.normal(mu, sigma, num).reshape(-1,1)
	y = np.random.normal(mu, sigma, num).reshape(-1,1)
	return x, y
