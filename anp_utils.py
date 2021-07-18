from matplotlib import pyplot as plt
import torch

def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.

  Args:
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains
        the x values of the context points.
    
    context_y: An array of shape [B,num_contexts,1] that contains
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
      """
    target_x = target_x[0].unsqueeze(0)
    target_y = target_y[0].unsqueeze(0)
    pred_y = pred_y[0].unsqueeze(0)
    std = std[0].unsqueeze(0)
    context_x = context_x[0].unsqueeze(0)
    context_y = context_y[0].unsqueeze(0)
    
    order = torch.argsort(target_x, dim=1)
    target_x = target_x.squeeze()[order]
    target_y = target_y.squeeze()[order]
    pred_y = pred_y.squeeze()[order]
    std = std.squeeze()[order]

    order = torch.argsort(context_x, dim=1)
    context_x = context_x.squeeze()[order]
    context_y = context_y.squeeze()[order]

    # print(target_x[0].shape, 
    #       target_y[0].shape,
    #       context_x[0].shape, 
    #       context_y[0].shape, 
    #       pred_y.shape, 
    #       std.shape)
    
    plt.plot(target_x[0], target_y[0], 'k.', markersize=2, label='target')
    plt.plot(target_x[0], pred_y[0], 'b.', markersize=2, label='predicted')
    plt.plot(context_x[0], context_y[0], 'g.', markersize=5, label='context')
    
    # plt.scatter(target_x[0], pred_y[0], s=2, color='blue', label='predicted')
    # plt.scatter(context_x[0], context_y[0], s=5, color='black', label='context')
    # plt.scatter(target_x[0], target_y[0], s=2, color='green', label='target')
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.25,
        facecolor='#65c9f7', #'blue',
        interpolate=True)

    # Make the plot pretty
    # plt.yticks([-6, 0, 6], fontsize=16)
    # plt.xticks([-6, 0, 6], fontsize=16)
    # plt.ylim([-6, 6])
    plt.legend()
    plt.grid('off')
    ax = plt.gca()
    return ax

def plot_functions_2d(target_x, target_y, context_x, context_y, pred_y, std, ax):
    target_x = target_x[0].unsqueeze(0)
    target_y = target_y[0].unsqueeze(0)
    pred_y = pred_y[0].unsqueeze(0)
    std = std[0].unsqueeze(0)
    context_x = context_x[0].unsqueeze(0)
    context_y = context_y[0].unsqueeze(0)

    # print(target_x.shape, 
    #       target_y.shape,
    #       context_x.shape, 
    #       context_y.shape, 
    #       pred_y.shape, 
    #       std.shape)
    
    order = torch.argsort(target_x, dim=1)[0,:,0]
    target_x = target_x[:, order, :]
    target_y = target_y[:, order, :]
    pred_y = pred_y[:, order, :]

    order = torch.argsort(context_x, dim=1)[0, :, 0]
    context_x = context_x[:, order, :]
    context_y = context_y[:, order, :]

    # Plot everything
    # print(target_x.shape, 
    #       target_y.shape,
    #       context_x.shape, 
    #       context_y.shape, 
    #       pred_y.shape, 
    #       std.shape)
        
    # plt.plot(target_x[0], target_y[0], 'k--', linewidth=2, label='target')
    # plt.plot(target_x[0], pred_y[0], 'b.', linewidth=1, label='predicted')
    # plt.plot(context_x[0], context_y[0], 'g.', markersize=5, label='context')
    
    ax.plot(target_x[0, :, 0], target_x[0, :, 1], pred_y[0, :, 0], 'b.', markersize=2, label='predicted')
    ax.plot(target_x[0, :, 0], target_x[0, :, 1], target_y[0, :, 0], 'k.', markersize=2, label='target')
    ax.plot(context_x[0, :, 0], context_x[0, :, 1], context_y[0, :, 0], 'g.', markersize=5, label='context')
    
    # ax.scatter(target_x[0, :, 0], target_x[0, :, 1], pred_y[0, :, 0], s=2, c='b', label='predicted')
    # ax.scatter(target_x[0, :, 0], target_x[0, :, 1], target_y[0, :, 0], s=2, c='k', label='target')
    # ax.scatter(context_x[0, :, 0], context_x[0, :, 1], context_y[0, :, 0], s=5, c='g', label='context')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    # Make the plot pretty
    # plt.yticks([-6, 0, 6], fontsize=16)
    # plt.xticks([-6, 0, 6], fontsize=16)
    # plt.ylim([-6, 6])
    
    # ax.set_xlim([-10,10])
    # ax.set_ylim([-10,10])
    # ax.set_zlim([0,30])
    ax.legend()
    ax.grid('off')
    
    return ax


def context_target_split(N, context_size):
    indices = torch.randperm(N)
    # indices = torch.linspace(0, N, N, dtype=torch.long)
    
    context_indices = indices[:context_size]
    target_indices = indices[context_size:]
    # print(context_indices, target_indices)
    return context_indices, target_indices

def compute_loss(posterior_dist, prior_dist, decoded_dist, target_y, sigma):
    # KL divergence
    kld = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean(dim=-1)
    
    # log probability
    log_prob = decoded_dist.log_prob(target_y).mean(dim=-1)

    # original proposed loss
    kld = kld[:, None].expand(log_prob.shape)
    loss = -log_prob + kld + 1.0*sigma.mean()
    return loss.mean(), kld, log_prob
    
    # l = torch.nn.MSELoss()
    # l = torch.nn.SmoothL1Loss()
    # loss = l(decoded_dist.mean, target_y)
    # mae_loss = torch.abs(decoded_dist.mean-target_y).sum()
    # loss = mae_loss    
    # return loss, kld, log_prob