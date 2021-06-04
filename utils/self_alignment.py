
import torch
import numpy as np
from utils.utils import checkParamIsSentToCuda

# self alignment-specific tension levels by chip
tension_level = {'R0402': (1000*500)/(400*200), 'R0603': (1000*500)/(600*300), 'R1005': (1000*500)/(1000*500)}

def self_alignment(inputs, model=None, toycfg=None):
    '''
    self alignment simulation for toy data

    options
        1. random (constant) angular and translational shift
        2. weighted shift by the relative distance between chip (Pre-AOI) and SPI in x,y directions
        3. weighted shift by the amount of rotation and translation of the chip (Pre-AOI) + volume..? + rotation angle (again..?) (post theta calculation...?)
    
    inputs
        [pre_x, pre_y, pre_theta, spi_x1, spi_y1, spi_x2, spi_y2, volume_1, volume_2, volume_difference]
    
    outputs
        [post_chip, post_theta]
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

''' 
    NOTE: BELOW contain 4 different self alignment demos
'''
def constantshift(inputs, toycfg):
    '''
    constant shift variables:
        position: random normal        
        total shift (dx, dy) = position (dx, dy)
    return
        [original (x,y) + total shift (dx, dy)]
    '''
    # input_info = ['pre_l', 'pre_w']
    # output_info = ['post_l', 'post_w']

    x_offset, x_noise, y_offset, y_noise = [dict.get(toycfg, variable) for variable in ['x_offset', 'x_noise', 'y_offset', 'y_noise']]
    pre_chip = inputs[:, 0:2]
    
    # # for statistics
    # xmax = torch.max(pre_chip[:,0])
    # xmin = torch.min(pre_chip[:,0])
    # ymax = torch.max(pre_chip[:,1])
    # ymin = torch.min(pre_chip[:,1])    
    # x_offset = ( xmax - xmin ) 
    # y_offset = ( ymax - ymin ) 

    x_offset = torch.FloatTensor([50])
    y_offset = torch.FloatTensor([50])

    post_chip = pre_chip + torch.tile(torch.FloatTensor([x_offset, y_offset]).to(inputs.device), (pre_chip.shape[0], 1))
    
    return post_chip


def shift(inputs, toycfg):
    '''
    constant shift variables:
        position: random normal        
        total shift (dx, dy) = position (dx, dy)
    return
        [original (x,y) + total shift (dx, dy)]
    '''
    # input_info = ['pre_l', 'pre_w']
    # output_info = ['post_l', 'post_w']

    x_offset, x_noise, y_offset, y_noise = [dict.get(toycfg, variable) for variable in ['x_offset', 'x_noise', 'y_offset', 'y_noise']]
    pre_chip = inputs[:, 0:2]
    x_offsets = torch.normal(mean=x_offset, std=x_noise, size=(pre_chip.shape[0], 1))
    y_offsets = torch.normal(mean=y_offset, std=y_noise, size=(pre_chip.shape[0], 1))
    offsets = torch.cat([x_offsets, y_offsets], -1)

    post_chip = pre_chip + offsets
    return post_chip

def shiftPolar(inputs, toycfg):
    '''
    constant shift variables:
        position: random normal
        angle: pre angle
        total shift (dx, dy) = position (dx, dy) + x,y components of angle
    return
        [original (x,y) + total shift (dx, dy)] 
    '''
    # input_info = ['pre_l', 'pre_w']
    # output_info = ['post_l', 'post_w']

    distance, noise = [dict.get(toycfg, variable) for variable in ['distance', 'noise']]
    num_samples = inputs.shape[0]
    pre_chip = inputs[:, 0:2]
    pre_angle = inputs[:, 2].unsqueeze(1)

    r_offsets = torch.normal(mean=distance, std=noise, size=(num_samples, 1))
    theta_offsets = torch.deg2rad(pre_angle)
    offsets = r_offsets * torch.cat([torch.cos(theta_offsets), torch.sin(theta_offsets)], -1)

    post_chip = pre_chip + offsets
    return post_chip

def tensionSimple(inputs, toycfg):
    '''
    simple self-alignment effect
        :demonstrates chip movement towards solder paste in the reflow oven
        :adds solder paste volume difference aspect (by multiplying to relative position difference in x)
        :adds resistance aspect for each chip
        :adds alpha for some additional variables not accounted for
    
    return  (= estimated chip position inspected at Post-AOI stage)
        [original (x,y) + shift (dx, dy), theta(all zeros)]
    '''
    # input_info = ['pre_l', 'pre_w','spi_x','spi_y', 'spi_vol_mean']
    # output_info = ['post_x', 'post_y']

    chipname = toycfg['chip']
    alpha = toycfg['alpha']
    num_samples = inputs.shape[0]

    pre_chip = inputs[:, 0:2]
    spi_center = inputs[:, 3:5]

    min_spi_vol = 0.7
    spi_vol_mean = torch.rand((num_samples,1)) * (1 - min_spi_vol) + min_spi_vol
    spi_vol_mean *= 100 # convert to percentage, not decimal
    inputs = torch.cat([pre_chip, spi_center, spi_vol_mean], -1)

    post_chip = pre_chip + (spi_center - pre_chip) * spi_vol_mean * tension_level[chipname] * alpha
    return post_chip

def tension(inputs, toycfg):
    input_info = ['pre_l', 'pre_w', 'pre_theta', 'spi_x1','spi_y1','spi_x2','spi_y2','spi_vol1','spi_vol2']
    output_info = ['post_l', 'post_w', 'post_theta']

    chipname = toycfg['chip']
    chipinfo = toycfg['chips'][chipname]
    length, width = chipinfo['length'], chipinfo['width']
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
    min_spi_vol = 0.7
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
    outputs = torch.cat([post_chip, post_theta], -1)
    return outputs





# for a toy problem setting
    
    # generate data
    #   (1) pre aoi (x,y)
    # inputs = torch.normal(mean=mu, std=sigma, size=(num_samples, 2))
    # #   (2) pre aoi (x,y) + spi (x,y) + spi volumes 1, 2 + spi volume difference
    # pre_chip = torch.normal(mean=mu, std=sigma, size=(num_samples, 2))
    # pre_theta = (torch.rand(size=(num_samples,1))*15) # pre angle range
    # spi_1 = pre_chip + \
    #     torch.rand(size=(num_samples, 2))* torch.FloatTensor([chip['length']*0.1, chip['width']*0.1]).repeat(num_samples, 1) # spi (x,y) are supposedly offset from pre (x,y) in the POSITIVE x direction here
    # spi_2 = pre_chip + \
    #     torch.rand(size=(num_samples, 2)) * torch.FloatTensor([-chip['length']*0.1, chip['width']*0.1]).repeat(num_samples, 1) # spi (x,y) are supposedly offset from pre (x,y) in the NEGATIVE x direction here    
    # volumes = torch.rand(size=(num_samples,2)) *  (1.0 - 0.7) + 0.7 # uniform(0, 10)
    # volume_difference = abs(volumes[:,0] - volumes[:,1]).unsqueeze(-1)
    # inputs = torch.cat([pre_chip, pre_theta, spi_1, spi_2, volumes, volume_difference], dim=1) # horizontally


    # if option == 1:
    #     '''
    #     constant shift variables:
    #         angle (deg): randn(mean=0, std=10)
    #         position: randn(mean=40, std=20)
            
    #         total shift (dx, dy) = position (x,y) * (x,y) components of the angle vector
            
    #     return
    #         [original (x,y) + shift (dx, dy), theta(all zeros)]
    #     '''
    #     direction = 0.
    #     d_noise = 10.
    #     theta_deg = torch.normal(mean=direction, std=d_noise, size=(inputs.shape[0], 1))
    #     theta_rad = torch.deg2rad(theta_deg)

    #     distance = 40.
    #     noise = 20.
    #     position_shift = torch.normal(mean=distance, std=noise, size=(inputs.shape[0], 2))
    #     position_shift *= torch.cat([torch.cos(theta_rad), torch.sin(theta_rad)], dim=1)

    #     post_chip = inputs[:, 0:2] + position_shift
    #     post_theta = torch.zeros(size=(post_chip.shape[0], 1))
    #     outputs = torch.cat([post_chip, post_theta], dim=1)
        
    # elif option == 2:
    #     '''
    #     simple self-alignment effect
    #         :demonstrates chip movement towards solder paste in the reflow oven
    #         :adds solder paste volume difference aspect (by multiplying to relative position difference in x)
    #         :adds resistance aspect for each chip
    #         :adds alpha for some additional variables not accounted for
        
    #     return  (= estimated chip position inspected at Post-AOI stage)
    #         [original (x,y) + shift (dx, dy), theta(all zeros)]
    #     '''
    #     # tension level by chip type
    #     tension_level = {'R0402': (1000*500)/(400*200), 'R0603': (1000*500)/(600*300), 'R1005': (1000*500)/(1000*500)}
    #     alpha = 0.1
        
    #     relative_spi = inputs[:, np.r_[3, 4]] - inputs[:, np.r_[0, 1]]
        
    #     # weigh relative spi
    #     weighted_relative_spi = relative_spi * torch.cat([(1 - inputs[:, -1]).unsqueeze(1), torch.ones(size=(inputs.shape[0], 1))], dim=1)
    #     weighted_relative_spi *= tension_level[chipname]
    #     weighted_relative_spi *= alpha

    #     # outs = []
    #     # for datum in inputs:
    #     #     pre_x, pre_y, _, SPI_x, SPI_y, _, _, _, _, SPI_diff = datum

    #     #     rel_SPI_x = (SPI_x - pre_x) * (1 - SPI_diff)
    #     #     rel_SPI_y = SPI_y - pre_y
                
    #     #     post_x = pre_x + rel_SPI_x * tension_level[chipname] * alpha
    #     #     post_y = pre_y + rel_SPI_y * tension_level[chipname] * alpha
    #     #     outs.append([post_x, post_y])
    #     # post_chip = torch.FloatTensor(np.array(outs))

    #     post_chip = inputs[:, 0:2] + weighted_relative_spi
    #     post_theta = torch.zeros(size=(post_chip.shape[0], 1))
    #     outputs = torch.cat([post_chip, post_theta], dim=1)

    # elif option == 3:
    #     # [pre_x, pre_y, pre_theta, spi_x1, spi_y1, spi_x2, spi_y2, volume_1, volume_2, volume_difference]
    #     # pre_theta_rad = torch.deg2rad(inputs[:, 2]) # pre angle
    
    #     relative_spi = inputs[:, np.r_[3,4,5,6]] - inputs[:, np.r_[0,1,0,1]]

    #     direction = 0.
    #     d_noise = 10.
    #     theta_deg = torch.normal(mean=direction, std=d_noise, size=(inputs.shape[0], 1))
    #     theta_rad = torch.deg2rad(theta_deg)

    #     tension_level = {'R0402': 50/8, 'R0603': 50/18, 'R1005': 1}
        
    #     alpha = 0.2
    #     beta = 0.0005
    #     post_chip = []
    #     post_theta = []
    #     for i in range(inputs.shape[0]):
    #         val = torch.FloatTensor([[relative_spi[i, 0], relative_spi[i, 2]],
    #                                 [relative_spi[i, 1], relative_spi[i, 3]],
    #                                 [1,1]])
    #         rotation_matrix = torch.FloatTensor([[torch.cos(theta_rad[i]), -torch.sin(theta_rad[i]), 0],
    #                                             [torch.sin(theta_rad[i]), torch.cos(theta_rad[i]), 0],
    #                                             [0, 0, 1]])
    #         translation_matrix = torch.FloatTensor([[-chip['length']/2, chip['length']/2],
    #                                                 [-chip['width']/2, chip['width']/2],
    #                                                 [0, 0]])
    #         val = rotation_matrix @ val - translation_matrix # 3x3 * 3x2 - 3x2 = 3x2
            
    #         volume_matrix = torch.FloatTensor([[-inputs[i, 7], inputs[i, 7]],
    #                                           [-inputs[i, 8], inputs[i, 8]]])
    #         val = val @ volume_matrix # 3x2 * 2x2 = 3x2
            
    #         tension = torch.sum(val, dim=0)[0:2] * tension_level[chipname] # 1x2

    #         post_x = inputs[i, 0] + alpha * tension[0] * torch.cos(theta_rad[i])
    #         post_y = inputs[i, 1] + alpha * tension[0] * torch.sin(theta_rad[i])
    #         post_chip.append([post_x, post_y])
    #         post_theta.append(inputs[i, 2] + torch.rad2deg(beta * tension[1]))
    #     post_chip = torch.FloatTensor(post_chip)
    #     post_theta = torch.FloatTensor(post_theta).unsqueeze(-1)
    #     outputs = torch.cat([post_chip, post_theta], dim=1)
    # return outputs