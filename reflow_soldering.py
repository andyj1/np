import torch
import numpy as np
import joblib
import os

# self alignment-specific tension levels by chip
tension_level = {'R0402': (1000*500)/(400*200), 'R0603': (1000*500)/(600*300), 'R1005': (1000*500)/(1000*500)}

model_dir = 'reflow_oven'
# file_path = 'regressor_R1005_50_trees_100_deep_random_forest.pkl'
file_path = 'R1005_0_20_20.pkl'
model_path = os.path.join(model_dir, file_path)
regressor = joblib.load(model_path)

def self_alignment(x1, x2=None):
    """
    inputs: either (x1) only or (x1 and x2)
        (1) x1 only: has x and y columns. **x1 can have more than 2 columns
        (2) x1 and x2: x1 has x and x2 has y    
    """
    method = 'shift1' # toycfg['method']
    # method = 'gt_approx'
    
    # match dimensions
    shifted_outputs = []
    if x2 is not None:
        try:
            x1 = torch.FloatTensor([x1]).unsqueeze(-1)
            x2 = torch.FloatTensor([x2]).unsqueeze(-1)
        except ValueError:
            pass
        x1 = torch.cat((x1, x2), dim=1)
    
    # for x1 input only, convert numpy ndarray to torch FloatTensor
    if type(x1) == np.ndarray:
        x1 = torch.FloatTensor(x1)
    assert type(x1) == torch.Tensor, type(x1)
    
    # apply self-alignment effect
    shifted_outputs = globals()[method](inputs=x1[:, 0:2])
    
    # if x1 input has more than 2 columns
    if x1.shape[1] > 2:
        shifted_outputs = torch.cat((shifted_outputs, x1[:, 2:]), dim=-1)
    
    # print('self alignment:', x1, '-->', shifted_outputs)
    # print(shifted_outputs.shape)
    return shifted_outputs, method

# ================ SHIFT IN X AND Y DIRECTIONS ONLY ================
def shift1(inputs):
    '''
    shift1: constant shift in x, y
        outputs = original (x,y) + random normal position offsets (dx, dy)
    '''
    
    SHIFT = (50, 50)
    x_offset = torch.FloatTensor([SHIFT[0]])
    y_offset = torch.FloatTensor([SHIFT[1]])
    
    xy_offsets = torch.FloatTensor([x_offset, y_offset])
    post_chip = inputs + torch.tile(xy_offsets, dims=(inputs.shape[0], 1)).to(inputs.device)

    outputs = post_chip
    return outputs


def shift2(inputs, toycfg):
    '''
    shift2: 
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
    #    post_theta = torch.normal(mean=angle_offset, std=angle_noise, size=(num_samples, 1))
    #    post_theta = post_theta.to(inputs.device)
    # ================================
    x_offsets = torch.normal(mean=x_offset, std=x_noise, size=(num_samples, 1))
    y_offsets = torch.normal(mean=y_offset, std=y_noise, size=(num_samples, 1))
    offsets = torch.cat([x_offsets, y_offsets], -1)
    
    offsets = offsets.to(inputs.device)
    
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

def shift3(inputs, toycfg):
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

def shift4(inputs, toycfg):
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

def gt_approx(inputs):
    '''
    gt_approx: MOM4 model using random forest regressor
        regression (dx, dy) according to random forest regressor
    return
        [regressed (x,y)]
    outputs = None
    '''
    global regressor
    outputs = regressor.predict(inputs)
    
    if isinstance(outputs, np.ndarray):
        outputs = torch.from_numpy(outputs)
    
    return outputs





if __name__ == '__main__':
    
    from create_self_alignment_model import customMOM4chipsample
    import pathlib
    import joblib
    import os
    import pandas as pd
    
    dir_path = pathlib.Path().absolute()
    file = './datasets/spi_clustered.csv'
    data_path = dir_path / file
    df = pd.read_csv(data_path, index_col=False).drop(['Unnamed: 0'], axis=1)
    df.reset_index(drop=True, inplace=True)
    assert df.isnull().sum().sum() == 0, 'there is a NULL value in the loaded data'
    
    chip = 'R1005'
    input_vars = ['PRE_L','PRE_W']
    test_size = 1000
    X_test = customMOM4chipsample(df, input_vars=input_vars, num_samples=test_size, chiptype=chip, random_state=42)
    print('='*10, f'using {test_size} samples', '='*10)
    print('chip: {chip}, input variables: {input_vars}')
    print('inputs:', X_test.shape)
    
    model_dir = 'reflow_oven'
    file_path = 'regressor_R1005_50_trees_100_deep_random_forest.pkl'
    model_path = os.path.join(model_dir, file_path)
    regressor = joblib.load(model_path)
    outputs, method = self_alignment(X_test, model=regressor)
    
    print('outputs:', outputs.shape)