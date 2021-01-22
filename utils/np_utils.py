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
    return [x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)]


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
