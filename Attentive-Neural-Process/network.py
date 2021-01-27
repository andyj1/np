from module import *
import torch

def kl_div(prior_mu, prior_var, posterior_mu, posterior_var):
    kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
    kl_div = 0.5 * kl_div.sum()
    return kl_div

class LatentModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """
    def __init__(self, num_hidden):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim=3)    # for z
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden)   # for r
        self.decoder = Decoder(num_hidden)
        self.BCELoss = nn.BCELoss()
        
    def forward(self, context_x, context_y, target_x, target_y=None):
        # print('context_x, context_y, target_x, target_y ', context_x.shape, context_y.shape, target_x.shape, target_y.shape)
        num_targets = target_x.size(1)

        # returns [mu,sigma] for the input data and [reparameterized sample from that distribution]
        # print(f'x shape: {context_x.shape}, y shape: {context_y.shape}')
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        
        # For training, update the latent vector
        posterior_mu, posterior_var = None, None
        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            z = posterior
        
        # For Generation keep the prior distribution as the latent vector
        else:
            z = prior

        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]
        
        # returns attention query for the encoder input after cross-attention
        r = self.deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]
        
        # prediction of target y given r, z, target_x
        y_pred = self.decoder(r, z, target_x)
        
        # For Training
        if target_y is not None:
            # get log probability
            bce_loss = self.BCELoss(torch.sigmoid(y_pred), target_y)
            
            # get KL divergence between prior and posterior
            kl = kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            
            # maximize prob and minimize KL divergence
            
            loss = bce_loss + kl
        
        # For Generation
        else:
            log_p = None
            kl = None
            loss = None
        
        # print('z:', z.shape, 'r:', r.shape, 'y_pred:',y_pred.shape, 'loss:', loss.item())
        
        return y_pred, kl, loss
    