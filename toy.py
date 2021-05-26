#!/usr/bin/env python3

import torch

class ToyData(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_samples = config['num_samples']
        # PRE L, W
        self.mu1 = config['mu1']
        self.sigma1 = config['sigma1']
        self.mu2 = config['mu2']
        self.sigma2 = config['sigma2']
        # PRE THETA
        self.mu_theta = config['mu_theta']
        self.sigma_theta = config['sigma_theta']
        # SPI L, W
        self.mu_spi1 = config['mu_spi1']
        self.mu_spi2 = config['mu_spi2']
        self.sigma_spi1 = config['sigma_spi1']
        self.sigma_spi2 = config['sigma_spi2']
        

    def preLW(self):
        l = torch.normal(mean=self.mu1, std=self.sigma1, size=(self.num_samples, 1))
        w = torch.normal(mean=self.mu2, std=self.sigma2, size=(self.num_samples, 1))
        return torch.cat([l, w], dim=1)

    def preAngle(self):
        return torch.normal(mean=self.mu_theta, std=self.sigma_theta, size=(self.num_samples, 1))

    def SPIcenter(self):
        l = torch.normal(mean=self.mu_spi1, std=self.sigma_spi1, size=(self.num_samples, 1))
        w = torch.normal(mean=self.mu_spi2, std=self.sigma_spi2, size=(self.num_samples, 1))
        return torch.cat([l, w], dim=1)

    def SPILW(self):
        pass

    def SPIVolumes(self):
        pass

if __name__=='__main__':
    import yaml
    with open('config.yml', 'r')  as file:
        cfg = yaml.load(file, yaml.FullLoader)
    toy = ToyData(cfg['toy'])
    inputs = torch.cat([toy.preLW(), toy.preAngle(), toy.SPIcenter()], dim=1)
    print(inputs.shape)

