#!/usr/bin/env python3

import torch
import yaml
import os

class ToyData(object):
    def __init__(self):
        
        config = yaml.load(open(os.path.join(os.getcwd(), 'toy/toyconfig.yml'), 'r'), yaml.FullLoader)['toy']
        self.num_samples = config['num_samples']
        
        seed = 42
        torch.manual_seed(seed)
        
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
        # SPI CENTER
        self.mu_spi_center1 = config['mu_spi_center1']
        self.mu_spi_center2 = config['mu_spi_center2']
        self.sigma_spi_center1 = config['sigma_spi_center1']
        self.sigma_spi_center2 = config['sigma_spi_center2']
    
    def preLW(self):
        l = torch.normal(mean=self.mu1, std=self.sigma1, size=(self.num_samples, 1)) - 50
        w = torch.normal(mean=self.mu2, std=self.sigma2, size=(self.num_samples, 1)) - 50
        return torch.cat([l, w], dim=1)

    def preAngle(self):
        return torch.normal(mean=self.mu_theta, std=self.sigma_theta, size=(self.num_samples, 1))

    def SPILW(self):
        l = torch.normal(mean=self.mu_spi1, std=self.sigma_spi1, size=(self.num_samples, 1))
        w = torch.normal(mean=self.mu_spi2, std=self.sigma_spi2, size=(self.num_samples, 1))
        return torch.cat([l, w], dim=1)
    
    def SPIVolumes(self):
        # generate SPI volume percentages in range [min, max]
        min_spi_vol_percentage = 0.70
        max_spi_vol_percentage = 1.00
        spi_vols = torch.rand(self.num_samples, 2) * (max_spi_vol_percentage - min_spi_vol_percentage) + min_spi_vol_percentage
        return spi_vols
    
    # ============================================================================
    def SPIcenter(self):
        l = torch.normal(mean=self.mu_spi_center1, std=self.sigma_spi_center1, size=(self.num_samples, 1))
        w = torch.normal(mean=self.mu_spi_center2, std=self.sigma_spi_center2, size=(self.num_samples, 1))
        return torch.cat([l, w], dim=1)

if __name__=='__main__':
    import yaml
    toy = ToyData()
    inputs = torch.cat([toy.preLW(), toy.preAngle(), toy.SPILW(), toy.SPIVolumes()], dim=1)
    print('='*10, 'toy data','='*10, )
    print(inputs.shape)

