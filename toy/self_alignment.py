
import torch
import numpy as np

# self alignment-specific tension levels by chip
tension_level = {'R0402': (1000*500)/(400*200), 'R0603': (1000*500)/(600*300), 'R1005': (1000*500)/(1000*500)}

import joblib
import os

model_dir = 'reflow_oven'
file_path = 'regressor_R1005_50_trees_100_deep_random_forest.pkl'
model_path = os.path.join(model_dir, file_path)
regressor = joblib.load(model_path)

def self_alignment(x1, x2=None):
    method = 'constantshift' #toycfg['method'] # shift / shiftPolar / tensionSimple / tension
    method = 'MOM4MODEL'
    
    shifted_outputs = []
    if x2 is not None:
        try:
            x1 = torch.FloatTensor([x1]).unsqueeze(-1)
            x2 = torch.FloatTensor([x2]).unsqueeze(-1)
        except ValueError:
            pass
        x1 = torch.cat((x1, x2), dim=1)
    
    if x1.shape[1] > 2:
        x1 = x1[:, 0:2]
        
    if type(x1) == np.ndarray:
        x1 = torch.FloatTensor(x1)
    assert type(x1) == torch.Tensor, type(x1)
    shifted_outputs = globals()[method](inputs=x1)
    
    if x1.shape[1] > 2:
        shifted_outputs = torch.cat((shifted_outputs, x1[:, 2:]), dim=-1)
    
    # print('self alignment:', x1, '-->', shifted_outputs)
    # print(shifted_outputs.shape)
    return shifted_outputs, method

# ================ SHIFT IN X AND Y DIRECTIONS ONLY ================
def constantshift(inputs):
    '''
    constant shift:
        total shift (dx, dy) = random normal position offsets (dx, dy)
    return
        [original (x,y) + total shift (dx, dy)]
    outputs = None
    '''
    
    SHIFT = (5, 5)
    x_offset = torch.FloatTensor([SHIFT[0]])
    y_offset = torch.FloatTensor([SHIFT[1]])
    
    xy_offsets = torch.FloatTensor([x_offset, y_offset])
    post_chip = inputs + torch.tile(xy_offsets, dims=(inputs.shape[0], 1)).to(inputs.device)

    # if inputs.shape[1] > 2: 
    #     outputs = torch.cat([post_chip, inputs[:,2:]], dim=-1)
    # else:
    #    outputs = post_chip
    outputs = post_chip
    return outputs

def MOM4MODEL(inputs):
    '''
    MOM4MODEL:
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