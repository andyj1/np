
import torch
import numpy as np
from utils.utils import checkParamIsSentToCuda


# self alignment-specific tension levels by chip
tension_level = {'R0402': (1000*500)/(400*200), 'R0603': (1000*500)/(600*300), 'R1005': (1000*500)/(1000*500)}

def self_alignment(inputs, model=None, toycfg=None):
    '''
    self alignment simulation for toy data:
        :assume inputs contain all available variables: [PRE L,W, SPI L,W, SPI volume, etc.]
        :newly generated data (e.g. SPI) are appended in the main loop
        : outputs

    options
        1. random (constant) angular and translational shift
        2. weighted shift by the relative distance between chip (Pre-AOI) and SPI in x,y directions
        3. weighted shift by the amount of rotation and translation of the chip (Pre-AOI) + volume..? + rotation angle (again..?) (post theta calculation...?)
    
    inputs
        [pre_L, pre_W, pre_theta, spi_x1, spi_y1, spi_x2, spi_y2, volume_1, volume_2, volume_difference, etc.]
    
    outputs
        [shifted inputs, other variables] (of same size as the inputs)
    '''

    # if using a model for MOM4 data
    if model is not None: 
        cuda_status = checkParamIsSentToCuda(inputs)
        if cuda_status == [True]: inputs = inputs.detach().cpu().numpy()
        shifted_outputs = model.predict(inputs) # evaluate using the self alignment model
        return shifted_outputs, None
    
    # else, toy problem setting
    if toycfg is None: raise Exception('Please input toy configurations.')

    # use toy self-alignment effect
    method = toycfg['method'] # shift / shiftPolar / tensionSimple / tension
    shifted_outputs = globals()[method](inputs, toycfg)
    return shifted_outputs, method

# ================ SHIFT IN X AND Y DIRECTIONS ONLY ================
def constantshift(inputs, toycfg=None):
    '''
    constant shift:
        total shift (dx, dy) = random normal position offsets (dx, dy)
    return
        [original (x,y) + total shift (dx, dy)]
    '''
    outputs = None
    
    pre_chip = inputs[:, 0:2]
    x_offset = torch.FloatTensor([50])
    y_offset = torch.FloatTensor([50])
    post_chip = pre_chip + torch.tile(torch.FloatTensor([x_offset, y_offset]).to(inputs.device), (pre_chip.shape[0], 1))
    
    if inputs.shape[1] > 2: 
        outputs = torch.cat([post_chip, inputs[:,2:]], dim=-1)
    else:
        outputs = post_chip
    return outputs

def shift(inputs, toycfg):
    '''
    shift:
        total shift (dx, dy) = random normal position offsets (dx, dy)
    return
        [original (x,y) + total shift (dx, dy)]
    '''
    outputs = None
    
    x_offset, x_noise, y_offset, y_noise = [dict.get(toycfg, variable) for variable in ['x_offset', 'x_noise', 'y_offset', 'y_noise']]
    num_samples = inputs.shape[0]
    x_offsets = torch.normal(mean=x_offset, std=x_noise, size=(num_samples, 1))
    y_offsets = torch.normal(mean=y_offset, std=y_noise, size=(num_samples, 1))
    xy_offsets = torch.cat([x_offsets, y_offsets], dim=-1)

    xy_offsets = xy_offsets.to(inputs.device)
    pre_chip = inputs[:, 0:2]
    post_chip = pre_chip + xy_offsets
    
    if inputs.shape[1] > 2: 
        outputs = torch.cat([post_chip, inputs[:,2:]], dim=-1)
    else:
        outputs = post_chip
    
    return outputs

# ================ SHIFT IN POSITIONS & ANGLES ================
def shiftPolar(inputs, toycfg):
    '''
    constant shift variables:
        position: random normal
        angle: pre angle
        total shift (dx, dy) = position (dx, dy) + x,y components of angle
    return
        [original (x,y) + total shift (dx, dy)] 
    '''
    outputs = None
    
    x_offset, x_noise, y_offset, y_noise, angle_offset, angle_noise = [dict.get(toycfg, variable) for variable in
                       ['x_offset', 'x_noise', 'y_offset', 'y_noise', 'angle_offset', 'angle_noise']]
    num_samples = inputs.shape[0]
    pre_chip = inputs[:, 0:2]
    pre_angle = inputs[:, 2].unsqueeze(1)
    
    # ================================
    # 1. total x,y offsets = X ~N(x_offset, x_noise), Y ~N(y_offset, y_noise)
    #
    # issue
    #   - not adding angular offset
    # ================================
    x_offsets = torch.normal(mean=x_offset, std=x_noise, size=(num_samples, 1))
    y_offsets = torch.normal(mean=y_offset, std=y_noise, size=(num_samples, 1))
    offsets = torch.cat([x_offsets, y_offsets], -1)
    
    offsets = offsets.to(inputs.device)
    # post_theta = torch.normal(mean=angle_offset, std=angle_noise, size=(num_samples, 1))
    # post_theta = post_theta.to(inputs.device)
    
    post_chip = pre_chip + offsets
    
    # ================================
    # 2. total x,y offsets = positional offset ~N(distance, noise) * x,y components(cos, sin) of PRE angle
    # issue
    #   - may scale largely by multiplying the positional and PRE angle's positional components
    #
    # ================================
    #
    # r_offsets = torch.normal(mean=xy_offset, std=xy_noise, size=(num_samples, 1))
    # theta_offsets = torch.deg2rad(pre_angle)
    #
    # offsets = r_offsets * torch.cat([torch.cos(theta_offsets), torch.sin(theta_offsets)], -1)
    # post_chip = pre_chip + offsets
    
    if inputs.shape[1] > 2: 
        outputs = torch.cat([post_chip, inputs[:,2:]], dim=-1)
    else:
        outputs = post_chip
    
    return outputs

def tensionSimple(inputs, toycfg):
    '''
        :demonstrates chip movement towards solder paste in proportion to the spi volume
        :hyperparameter alpha
        :tension level by chip    
    '''
    outputs = None
    
    chipname = toycfg['chip']
    alpha = toycfg['alpha']
    num_samples = inputs.shape[0]

    pre_chip = inputs[:, 0:2]
    
    # ================================t
    spi_center = (torch.normal(mean=toycfg['mu_spi1'], std=toycfg['sigma_spi1'], size=(pre_chip.shape[0], 2)).to(inputs.device)
                + torch.normal(mean=toycfg['mu_spi2'], std=toycfg['sigma_spi2'], size=(pre_chip.shape[0], 2)).to(inputs.device))/2

    min_spi_vol = 0.7
    
    spi_vol_mean = torch.mean(torch.rand((num_samples,2)) * (1 - min_spi_vol) + min_spi_vol, dim= -1)
    spi_vol_mean *= 100 # convert to percentage, not decimal
    post_chip = pre_chip + (spi_center - pre_chip) * spi_vol_mean.reshape(-1,1).to(inputs.device) * tension_level[chipname] * alpha
    

    # chipname = toycfg['chip']
    # alpha = toycfg['alpha']
    # num_samples = inputs.shape[0]
    #
    # pre_chip = inputs[:, 0:2]
    # spi_center = inputs[:, 3:5]
    #candidate_outputs.shape
    # min_spi_vol = 0.7
    # spi_vol_mean = torch.rand((num_samples,1)) * (1 - min_spi_vol) + min_spi_vol
    # spi_vol_mean *= 100 # convert to percentage, not decimal
    # inputs = torch.cat([pre_chip, spi_center, spi_vol_mean], -1)
    #
    # post_chip = pre_chip + (spi_center - pre_chip) * spi_vol_mean * tension_level[chipname] * alpha
    
    if inputs.shape[1] > 2: 
        outputs = torch.cat([post_chip, inputs[:,2:]], dim=-1)
    else:
        outputs = post_chip
    return outputs

def tension(inputs, toycfg):
    outputs = None

    chipname = toycfg['chip']
    length = toycfg['chips'][chipname]['length']
    width = toycfg['chips'][chipname]['width']
    alpha = toycfg['alpha']
    beta = toycfg['beta']
    num_samples = inputs.shape[0]

    # mu, sigma : length(x) offset | mu2, sigma2 : width(y) offset
    pre_chip = inputs[:, 0:2]
    pre_angle = inputs[:, 2].unsqueeze(1)
    spi_center = inputs[:, 3:5]

    chipL_half = torch.FloatTensor([length/2, 0])
    spi1 = spi_center - chipL_half # num_samples x 2(spi_x1, spi_y1)
    spi2 = spi_center + chipL_half  # num_samples x 2(spi_x2, spi_y2)
    min_spi_vol = 0.70 # percentage
    spi_vol = torch.rand((num_samples,2)) * (1 - min_spi_vol) + min_spi_vol # num_samples x 2(spi_vol1, spi_vol2)
    spi_vol *= 100 # convert to percentage, not decimal

    inputs = torch.cat([pre_chip, pre_angle, spi1, spi2, spi_vol], dim=-1)

    chip_left_side = pre_chip + torch.cat([length/2 * torch.cos(torch.deg2rad(-pre_angle + 180)),
                                           length/2 * torch.sin(torch.deg2rad(-pre_angle + 180))], dim=-1)
    chip_right_side = pre_chip + torch.cat([length/2 * torch.cos(torch.deg2rad(-pre_angle)),
                                            length/2 * torch.sin(torch.deg2rad(-pre_angle))], dim=-1)
    chip2spi1 = spi1 - chip_left_side # spi1 vector
    chip2spi2 = spi2 - chip_right_side # spi2 vector
    transpose = chip2spi1 * spi_vol[:,0].unsqueeze(-1) + chip2spi2 * spi_vol[:,1].unsqueeze(-1)  # x_offset, y_offset

    rotation = - torch.linalg.norm(chip2spi1) * torch.cos(torch.deg2rad(-pre_angle)) * spi_vol[:,0].unsqueeze(-1) \
                + torch.linalg.norm(chip2spi2) * torch.cos(torch.deg2rad(-pre_angle)) * spi_vol[:,1].unsqueeze(-1)

    post_chip = pre_chip + alpha * tension_level[chipname] * transpose
    post_theta = pre_angle + beta * tension_level[chipname] * rotation
    
    if inputs.shape[1] > 2: 
        outputs = torch.cat([post_chip, inputs[:,2:]], dim=-1)
    else:
        outputs = post_chip
        
    return outputs
