import torch
import warnings
import gc

def setup():
    torch.cuda.empty_cache()
    gc.collect()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    
def objective(x, y):
    
    
    return 