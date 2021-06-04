import time
import argparse

def parse_args():
    parse_start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', help='sets to toy dataset')
    parser.add_argument('--load_rf', default='reflow_oven/models_2var/regressor_R1005_50_trees_100_deep_random_forest.pkl', type=str, help='path to reflow oven model')
    parser.add_argument('--model', default='GP', type=str, help='surrogate model type')
    parser.add_argument('--load', default=None, type=str, help='path to checkpoint [pt] file')
    parser.add_argument('--chip', default=None, type=str, help='chip part type')
    # parser.add_argument('--not_increment_context', default=True, action='store_false', help='increments context size over iterations, instead of target size')
    parser.add_argument('--cholesky', default=False, action='store_true', help='sets boolean to use cholesky decomposition')
    args = parser.parse_args()
    
    parse_end = time.time(); 
    print('[INFO] parsing arguments: %.3f seconds' % (parse_end - parse_start))
    return args