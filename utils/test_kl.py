import torch
import numpy as np

''' 1 '''
# https://github.com/soobinseo/Attentive-Neural-Process/blob/master/network.py
def kl_div(prior_mu, prior_var, posterior_mu, posterior_var):
    kl_div = (torch.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
    kl_div = 0.5 * kl_div.sum()
    return kl_div

''' 2 '''
# equation (ans2)
def kullback_leiber_divergence(mu, sigma, nu, tau):
    sigma = sigma + 1e-16
    tau = tau + 1e-16
    return ((sigma**2 + (mu - nu )**2) / tau**2 - 1 + 2 * torch.log(tau / sigma)).sum() / 2

a,b,c,d = torch.FloatTensor([0.]), torch.FloatTensor([1.]), torch.FloatTensor([1.]), torch.FloatTensor([2.])

''' 3 '''
# tensorflow version (ans3)
def kl_div_tf(a,b,c,d):
    import tensorflow as tf
    distribution_a = tf.compat.v1.distributions.Normal(loc=a, scale=b)
    distribution_b = tf.compat.v1.distributions.Normal(loc=c, scale=d)
    kl_div = tf.compat.v1.distributions.kl_divergence(distribution_a, distribution_b)
    return kl_div

''' 4 '''
# torch version (ans4)
from torch.distributions.kl import kl_divergence
def KLD_gaussian(q_target, q_context):
    return kl_divergence(q_target, q_context).mean(dim=0).sum()
dist1 = torch.distributions.normal.Normal(loc=a, scale=b)
dist2 = torch.distributions.normal.Normal(loc=c, scale=d)

ans1 = kl_div(a,b,c,d)
ans2 = kullback_leiber_divergence(a,b,c,d)
ans3 = kl_div_tf(a,b,c,d)
ans4 = KLD_gaussian(dist1, dist2)



# other
# https://github.com/jusonn/Neural-Process/blob/master/np_mnist.py
def kl_div_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    var_q = torch.exp(logvar_q)
    kl_div = (var_q + (mu_q - mu_p)**2) / var_p - 1.0 + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum()
    return kl_div


# ans1 is off
# ans2,3,4 are all equivalent implementations
print(ans1, ans2, ans3, ans4)
