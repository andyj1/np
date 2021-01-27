#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys

class DeterministicEncoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i
    
    >> determinisitic path
    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    y_dim : int
        Dimension of y values.
    h_dim : int
        Dimension of hidden layer.
    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(DeterministicEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)
        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class LatentEncoder(nn.Module):
    """
    Maps (x, y) pairs into the mu and sigma parameters defining the normal
    distribution of the latent variables z.
    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)
    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)
    """
    def __init__(self, r_dim, z_dim, h_dim, x_dim, y_dim):
        super(LatentEncoder, self).__init__()
    
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

        self.deterministic_encoder = DeterministicEncoder(self.x_dim, self.y_dim, self.h_dim, self.r_dim)
    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.
        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)
    
    def forward(self, x, y):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        
        r_i_flat = self.deterministic_encoder(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
                
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma
        
        
        
        
class Decoder(nn.Module):
    """
    Decoder for maping [target_x] and latent [z] samples to [target_y]
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        layers = [nn.Linear(input_dim + latent_dim, hidden_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(hidden_dim, hidden_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(hidden_dim, hidden_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(hidden_dim, output_dim)
        self.hidden_to_sigma = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, target_x, z_sample):
        '''
        target_x: [batch_size, target_x_size, input_dim]
        z_sample: [batch_size, latent_dim]
        '''
        batch_size, num_points, _ = target_x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        
        z = z_sample.unsqueeze(1).repeat(1, num_points, 1)
        
        # Flatten x and z to fit with linear layer
        x_flat = target_x.view(batch_size * num_points, self.input_dim)
        z_flat = z.view(batch_size * num_points, self.latent_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.output_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.output_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * nn.functional.softplus(pre_sigma)
        
        # make distribution
        loc = mu.squeeze()
        scale = sigma.squeeze()
        mvn_dist = torch.distributions.MultivariateNormal(loc, scale_tril=torch.diag(scale))
        
        
        return mvn_dist, mu, sigma
