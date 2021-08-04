import random
import numpy as np

def simple(points, dist_mu = 10, dist_sigma = 10, degree_mu = 0, degree_sigma = 30, hv = True) :
	num = points.shape[0]
	r_dist = np.random.normal(dist_mu, dist_sigma, num)
	r_degree = np.random.normal(degree_mu, degree_sigma, num)
	if hv :
		return points[:,0] + r_dist * np.cos(r_degree * np.pi / 180)
	else :
		return points[:,0] + r_dist * np.sin(r_degree * np.pi / 180)