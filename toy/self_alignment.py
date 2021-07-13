
import torch
import numpy as np

# self alignment-specific tension levels by chip
tension_level = {'R0402': (1000*500)/(400*200), 'R0603': (1000*500)/(600*300), 'R1005': (1000*500)/(1000*500)}

def self_alignment(inputs):
    method = 'constantshift' #toycfg['method'] # shift / shiftPolar / tensionSimple / tension
    shifted_outputs = globals()[method](inputs)
    return shifted_outputs, method

# ================ SHIFT IN X AND Y DIRECTIONS ONLY ================
def constantshift(inputs):
    '''
    constant shift:
        total shift (dx, dy) = random normal position offsets (dx, dy)
    return
        [original (x,y) + total shift (dx, dy)]
    '''
    outputs = None
    
    x_offset = torch.FloatTensor([50])
    y_offset = torch.FloatTensor([50])
    
    pre_chip = inputs[:, 0:2] 
    xy_offsets = torch.FloatTensor([x_offset, y_offset]).to(inputs.device)
    post_chip = pre_chip + torch.tile(xy_offsets, dims=(pre_chip.shape[0], 1))

    if inputs.shape[1] > 2: 
        outputs = torch.cat([post_chip, inputs[:,2:]], dim=-1)
    else:
        outputs = post_chip
    return outputs
