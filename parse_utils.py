import argparse
import configparser

def parse():
    # config.ini
    cfg = configparser.ConfigParser()
    cfg.read('config.ini')
    train_cfg = dict(zip([key for key, _ in cfg.items('train')], \
                         [int(val) if val.isdigit() \
                          else bool(val) if val=='yes' or val=='no' \
                          else [int(n.strip()) for n in val[1:-1].split(',')] if val[0]=='[' and val[-1]==']' \
                          else float(val) if '.' in val \
                          else val for _, val in cfg.items('train')]))
    
    data_cfg = dict(zip([key for key, _ in cfg.items('custom_data')], 
                        [float(val) for _, val in cfg.items('custom_data')]))
    
    acq_cfg = dict(zip([key for key, _ in cfg.items('acquisition')], 
                       [[int(n.strip()) for n in val[1:-1].split(',')] if val[0]=='[' and val[-1]==']' \
                        else int(val) if val.isdigit() \
                        else float(val) \
                        for _, val in cfg.items('acquisition')]))
    
    # system level argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='sine', type=str, help='specify a dataset: [sine or parabola]')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help='by default, all outputs are suppressed')
    args = parser.parse_args()
    
    # modifications / conditions to cfg (since only cfg is carried along in the process)
    if args.dataset == 'sine': train_cfg['x_dim'] = 1
    elif args.dataset == 'parabola': assert train_cfg['x_dim'] in [1, 2]
    train_cfg['verbose'] = args.verbose
    acq_cfg['input_dim'] = train_cfg['x_dim']    
    train_cfg['dataset'] = args.dataset
    
    return train_cfg, data_cfg, acq_cfg, args


