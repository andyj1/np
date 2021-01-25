#!/usr/bin/env python3

import time
import joblib
import numpy as np

def checkParamIsSentToCuda(args):
    status = []
    for i, arg in enumerate(args):
        status.append(arg.is_cuda)
    return status

def loadReflowOven(args, chip='R1005', inputtype=0):
    # load RF regressor 
    print('='*10, 'loading regressor model:', end='')
    loadRFRegressor_start = time.time()
    
    '''
    # chip: chip type for selecting the model
    # inputtype: PRE or PRE-SPI for (x,y) positions
    
    # select a reflow oven model
    # model = -1
    # if chip == 'R0402' and inputtype == 0:
    #     model = 0
    # elif chip == 'R0402' and inputtype == 1:
    #     model = 1
    # elif chip == 'R0603' and inputtype == 0:
    #     model = 2
    # elif chip == 'R0603' and inputtype == 1:
    #     model = 3
    # elif chip == 'R1005' and inputtype == 0:
    #     model = 4
    # elif chip == 'R1005' and inputtype == 1:
    #     model = 5
    # elif chip == 'all' and inputtype == 0:
    #     model = 6
    # elif chip == 'all' and inputtype == 1:
    #     model = 7
    
    # override for the case: each chip data with all chip data trained reflow oven model
    # model = 6 # PRE
    # # model = 7 # PRE-SPI

    # model_paths = ['model1', ... ]
    '''

    # reflow oven maps [PRE] to POST 
    reflow_oven_model_path = args.load_rf
    regr_multirf = joblib.load(reflow_oven_model_path)

    loadRFRegressor_end = time.time()
    print(': took: %.3f seconds' % (loadRFRegressor_end - loadRFRegressor_start))
    return regr_multirf

# currently only takes into account positions in the x and y direction
def objective(x, y):
    return np.linalg.norm((x,y))    

'''
reflow_oven: function to model reflow oven shift behavior 
             from MultiOutput Random Forest Regressor
'''
def reflow_oven(inputs, model):
    # download from GPU
    if inputs.is_cuda:
        inputs = inputs.cpu()

    # evaluate
    outputs = model.predict(inputs)
    
    return outputs
