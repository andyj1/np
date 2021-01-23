import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.kl import kl_divergence
from numpy import arange, argmax, asarray, vstack

# utility functions
def log_likelihood(mu, std, target):
    mu = mu.unsqueeze(dim=0)
    std = std.unsqueeze(dim=0)
    norm = torch.distributions.Normal(mu, std)
    target = target.unsqueeze(dim=1)
    return norm.log_prob(target).sum(dim=0).mean()


def KLD_gaussian(q_target, q_context):
    return kl_divergence(q_target, q_context).mean(dim=0).sum()


def random_split_context_target(x, y, n_context):
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    x = x.cpu()
    y = y.cpu()
    context_x, context_y, target_x, target_y = [x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)]
    
    return context_x, context_y, target_x, target_y


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

# ===== ANP utils =====

def collate_fn(batch):
    # Puts each data field into a tensor with outer dimension batch size
    assert isinstance(batch[0], tuple)
        
    max_num_context = 784
    num_context = np.random.randint(10,784) # extract random number of contexts
    num_target = np.random.randint(0, max_num_context - num_context)
    num_total_points = num_context + num_target # this num should be # of target points
#     num_total_points = max_num_context
    context_x, context_y, target_x, target_y = list(), list(), list(), list()
    
    for d, _ in batch:
        total_idx = np.random.choice(range(784), num_total_points, replace=False)
        total_idx = list(map(lambda x: (x//28, x%28), total_idx))
        c_idx = total_idx[:num_context]
        c_x, c_y, total_x, total_y = list(), list(), list(), list()
        for idx in c_idx:
            c_y.append(d[:, idx[0], idx[1]])
            c_x.append((idx[0] / 27., idx[1] / 27.))
        for idx in total_idx:
            total_y.append(d[:, idx[0], idx[1]])
            total_x.append((idx[0] / 27., idx[1] / 27.))
        c_x, c_y, total_x, total_y = list(map(lambda x: torch.FloatTensor(x), (c_x, c_y, total_x, total_y)))
        context_x.append(c_x)
        context_y.append(c_y)
        target_x.append(total_x)
        target_y.append(total_y)
        
        
    context_x = torch.stack(context_x, dim=0)
    context_y = torch.stack(context_y, dim=0).unsqueeze(-1)
    target_x = torch.stack(target_x, dim=0)
    target_y = torch.stack(target_y, dim=0).unsqueeze(-1)
    
    return context_x, context_y, target_x, target_y


def kl_div(prior_mu, prior_var, posterior_mu, posterior_var):
    kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
    kl_div = 0.5 * kl_div.sum()
    return kl_div
