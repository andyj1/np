
import torch
import numpy as np

# self alignment-specific tension levels by chip
tension_level = {'R0402': (1000*500)/(400*200), 'R0603': (1000*500)/(600*300), 'R1005': (1000*500)/(1000*500)}

def self_alignment(x1, x2=None):
    method = 'constantshift' #toycfg['method'] # shift / shiftPolar / tensionSimple / tension
    
    shifted_outputs = []
    if x2 is not None and isinstance(x1, np.float64):
        x1 = torch.FloatTensor([x1]).unsqueeze(1)
        x2 = torch.FloatTensor([x2]).unsqueeze(1)
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
    
    x_offset = torch.FloatTensor([5])
    y_offset = torch.FloatTensor([5])
    
    xy_offsets = torch.FloatTensor([x_offset, y_offset])
    post_chip = inputs + torch.tile(xy_offsets, dims=(inputs.shape[0], 1))

    # if inputs.shape[1] > 2: 
    #     outputs = torch.cat([post_chip, inputs[:,2:]], dim=-1)
    # else:
    #    outputs = post_chip
    outputs = post_chip
    return outputs
