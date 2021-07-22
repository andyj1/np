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
    
    data_cfg = dict(zip([key for key, _ in cfg.items('data')], 
                        [float(val) for _, val in cfg.items('data')]))
    
    acq_cfg = dict(zip([key for key, _ in cfg.items('acquisition')], 
                       [[int(n.strip()) for n in val[1:-1].split(',')] if val[0]=='[' and val[-1]==']' \
                        else int(val) if val.isdigit() \
                        else float(val) \
                        for _, val in cfg.items('acquisition')]))
    
    # system level argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default=None, type=str, help='specify a dataset: [sine or parabola]')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help='by default, all outputs are suppressed')
    parser.add_argument('--visualize' , '-vis', default=False, action='store_true', help='turns on visualization during training')
    parser.add_argument('--random_state' ,'-rs', default=1, type=int, help='random state')
    parser.add_argument('--num_workers', '-ns', default=4, type=int, help='number of process workers')
    parser.add_argument('--num_samples', default=0, type=int, help='number of samples')
    args = parser.parse_args()
    
    # modifications / conditions to cfg (since only cfg is carried along in the process)
    if args.dataset == 'sine': train_cfg['x_dim'] = 1
    elif args.dataset == 'parabola': assert train_cfg['x_dim'] in [1, 2]
    train_cfg['verbose'] = args.verbose
    train_cfg['visualize'] = args.visualize
    acq_cfg['input_dim'] = train_cfg['x_dim']    
    train_cfg['dataset'] = args.dataset
    train_cfg['random_state'] = args.random_state
    train_cfg['num_workers'] = args.num_workers
    if args.num_samples != 0:
        train_cfg['num_samples'] = args.num_samples
    
    cfg = dict({'train_cfg': train_cfg, 'data_cfg': data_cfg, 'acq_cfg': acq_cfg})
    
    return cfg, args


