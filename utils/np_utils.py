#!/usr/bin/env python3

import os

import numpy as np
import torch
from torch.distributions.kl import kl_divergence

def log_likelihood(mu, sigma, target):
    target = target.squeeze(0)
    target = target.unsqueeze(dim=1)
    mu = mu.unsqueeze(dim=0)
    sigma = sigma.unsqueeze(dim=0)
    '''
    [LOG LIKELIHOOD] target: torch.Size([num_target, 1, 1]) 
                        mu: torch.Size([1, num_target, z_dim*2, 1]) 
                        sigma: torch.Size([1, num_target, z_dim*2, 1])
    '''
    # print('target:', target.shape, 'mu:',mu.shape, 'sigma:',sigma.shape)
    
    # norm = torch.distributions.Normal(mu.squeeze(0), sigma.squeeze(0))
    # return norm.log_prob(target).sum(dim=0).mean()

    # equation: https://www.statlect.com/glossary/log-likelihood
    return -(target - mu)**2 / (2 * sigma**2) - torch.log(sigma)

def kl_div(q_target, p_context):
    ''' example
    posterior = torch.distributions.normal.Normal(loc=mu, scale=sigma)
    prior = torch.distributions.normal.Normal(loc=mu, scale=sigma)
    kl_div(posterior, prior) := KLD(posterior || prior)
    '''
    return kl_divergence(q_target, p_context).mean(dim=0).sum()

def random_split_context_target(x, y, n_context):
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    x = x.cpu()
    y = y.cpu()
    context_x, context_y, target_x, target_y = [x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)]
    
    # alternative
    # context_idx = random.sample(range(train_X.shape[0]), cfg['np']['num_context'])
    # target_idx = np.delete(range(train_X.shape[0]), context_idx)
    # x_context, x_target = train_X[context_idx], train_X[target_idx]
    # y_context, y_target = train_Y[context_idx], train_Y[target_idx]
    return context_x, context_y, target_x, target_y

# reparam trick for tractability in randomness in the input
# def reparametrize(mu, logvar, num_target):
#     std = torch.exp(0.5 * logvar)
#     eps = torch.randn_like(std)
#     z_sample = eps.mul(std).add_(mu)
#     z_sample = z_sample.unsqueeze(1).expand(-1, num_target, -1)
#     return z_sample



'''
def visualize(x, y, x_star, model, epoch, xvar='x', yvar='y', result_path='./'):
    r_z = model.data_to_r(x, y)
    z_mu, z_std = model.r_to_z(r_z)
    zsamples = model.reparametrize(z_mu, z_std, 3)
    mu, sigma = model.decoder(x_star, zsamples)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(mu.size(1)):
        ax.plot(x_star.data.cpu().numpy(),
                mu[:, i].data.cpu().numpy(), linewidth=1)
        ax.fill_between(
            x_grid[:, 0].data.cpu().numpy(), (mu[:, i] - sigma[:, i]
                                              ).detach().cpu().numpy(),
            (mu[:, i] + sigma[:, i]).detach().cpu().numpy(), alpha=0.2
        )
        ax.scatter(x.data.cpu().numpy(), y.data.cpu().numpy(), color='b')
        # ax.plot(all_x_np, all_y_np, color='b') # plot all points for comparison
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
    # plt.pause(0.0001)
    fig.canvas.draw()
    fig.savefig(os.path.join(
        result_path, f'{parttype}_{xvar}_{yvar}_epoch_{str(epoch)}.png'))
    plt.show()
'''
