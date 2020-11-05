import torch
import torch.nn as nn
from torch.nn import functional as F

from np_utils import (KLD_gaussian, log_likelihood,
                      random_split_context_target, visualize)


class NeuralProcessRegressor(nn.Module):
    # class definition
    def __init__(self, hidden_dim, decoder_dim, z_samples):
        super(NeuralProcessRegressor, self).__init__()
        in_dim = 2
        out_dim = 2
        self.z_dim = 2
        self.z_samples = z_samples
        # for data (Xc, Yc) --> representation (r)
        self.hidden1 = nn.Linear(in_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, out_dim)
        # for representation (r) --> latent vector (z)
        self.r_to_z_mu = nn.Linear(in_dim, 1)
        self.r_to_z_std = nn.Linear(in_dim, 1)

        self.decoder1 = nn.Linear(in_dim+1, decoder_dim)
        self.decoder2 = nn.Linear(decoder_dim, out_dim)

        nn.init.normal_(self.hidden1.weight)
        nn.init.normal_(self.hidden2.weight)
        nn.init.normal_(self.decoder1.weight)
        nn.init.normal_(self.decoder2.weight)

    # data to representations, aggregated
    def data_to_r(self, x, y):
        x_y = torch.cat([x, y], dim=1)
        hidden1 = self.hidden1(x_y)
        hidden1_activated = F.relu(hidden1)
        r_i = self.hidden2(hidden1_activated)

        # mean aggregate
        r = r_i.mean(dim=0)
        return r

    # representation to latent vector
    def r_to_z(self, r):
        mean = self.r_to_z_mu(r)
        log_var = self.r_to_z_std(r)
        return mean, F.softplus(log_var)

    # reparam trick for tractability in randomness in the input
    def reparametrize(self, mu, std, n):
        eps = torch.autograd.Variable(std.data.new(n, self.z_dim).normal_())
        z = eps.mul(std).add_(mu)
        return z

    # decoder
    def decoder(self, x_pred, z):
        z = z.unsqueeze(-1).expand(z.size(0), z.size(1),
                                   x_pred.size(0)).transpose(1, 2)
        x_pred = x_pred.unsqueeze(0).expand(
            z.size(0), x_pred.size(0), x_pred.size(1))
        x_z = torch.cat([x_pred, z], dim=-1)

        decoded1 = self.decoder1(x_z).squeeze(-1).transpose(0, 1)
        decoder1_activated = torch.sigmoid(decoded1)
        decoded2 = self.decoder2(decoder1_activated)

        mu, logstd = torch.split(decoded2, 1, dim=-1)
        mu = mu.squeeze(-1)
        logstd = logstd.squeeze(-1)
        std = F.softplus(logstd)
        return mu, std

    def forward(self, x_context, y_context, x_target, y_target):
        # make x, y stack
        x_all = torch.cat([x_context, x_target], dim=0)
        y_all = torch.cat([y_context, y_target], dim=0)

        # context data -> context repr
        r = self.data_to_r(x_context, y_context)    # data -> repr
        z_mean, z_std = self.r_to_z(r)              # repr -> latent z

        # all data -> global repr
        r_all = self.data_to_r(x_all, y_all)
        z_mean_all, z_std_all = self.r_to_z(r_all)

        # reparameterize
        zs = self.reparametrize(z_mean_all, z_std_all, self.z_samples)

        # decoder
        mu, std = self.decoder(x_context, zs)
        return mu, std, z_mean_all, z_std_all, z_mean, z_std

    # if not os.path.exists('./results'):
    #     os.makedirs('./results')
    def train(epochs, n_context, all_x_np, all_y_np):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            # split into
            x_context, y_context, x_target, y_target = random_split_context_target(
                all_x_np, all_y_np, n_context)
            # send to gpu
            x_context = torch.from_numpy(x_context).to(device)
            y_context = torch.from_numpy(y_context).to(device)
            x_target = torch.from_numpy(x_target).to(device)
            y_target = torch.from_numpy(y_target).to(device)
            # forward pass
            mu, std, z_mean_all, z_std_all, z_mean, z_std = model(
                x_context, y_context, x_target, y_target)
            # loss calculation
            loss = -log_likelihood(mu, std, y_context) + \
                KLD_gaussian(z_mean_all, z_std_all, z_mean, z_std)
            # backprop
            loss.backward()
            training_loss = loss.item()
            optimizer.step()
            print('epoch: {} loss: {}'.format(epoch, training_loss/200))

            # save figure at every 100th training epoch
            if epoch % 1000 == 0:
                visualize(x_context, y_context, torch.from_numpy(vals.reshape(-1, 1).astype(np.float32)).to(device),
                          model, epoch, xvar, yvar, result_path='./results')

            # test with target points
            # TODO
