
import argparse
import os
import pathlib
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

pd.set_option('display.max_columns', None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse():
    print('='*10,'parsing','='*10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--train_data',default='pre_all', help='[pre_all, pre_chip, pre-spi_chip]')
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--test_size', default=100, type=int)
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--max_depth', default=10)
    parser.add_argument('--num_predictors', default=10)
    args = parser.parse_args()
    
    # usage: main.py [--toy] [--np] [--load path/to/checkpoint]
    return args

''' choosing the number of predictors for random forest
- if the number of observatiosn is large, but the number of trees is too small, 
then some observations will be predicted only once or even not at all
- if the number of predictors is large but the number of trees is too small, then some features can theoretically be missed in all subspaces used
- both cases (above) result in the decrease of RF predictive power

According to [Link](https://mljar.com/blog/how-many-trees-in-random-forest/), for a dataset of 1e5 rows, 1e3 RF predictors has a strong dependency
'''

if __name__ == '__main__':
    # parse arguments
    args = parse()

    if args.test_size == 0: 
        args.test = False
            
    # set data path
    dir_path = pathlib.Path().absolute()
    file = '../data/imputed_data.csv'
    data_path = os.path.join(dir_path, file)

    # set result path
    result_path = './models'
    if not os.path.isdir(result_path):
        os.makedirs(result_path, mode=0o755, exist_ok=True)

    ''' load '''
    # read from dataset
    df = pd.read_csv(data_path  ,index_col=False).drop(['Unnamed: 0'], axis=1)
    df.reset_index(drop=True, inplace=True)
    assert df.isnull().sum().sum() == 0

    # set random state for train-test-split
    SEED = np.random.RandomState(42)

    # set train data
    X, y, chip, train_save_filename = None, None, None, None
    if args.train_data == 'pre_all':
        X = df[['PRE_L','PRE_W']].to_numpy()
        y = df[['POST_L','POST_W']].to_numpy()
    elif args.train_data == 'pre_chip':
        chip = 'R0402'                      # ['R0402','R0603','R1005]
        df = df[df['PartType']==chip]
        X =  df[['PRE_L','PRE_W']].to_numpy()
        y =  df[['POST_L','POST_W']].to_numpy()
    elif args.train_data == 'pre-spi_chip':
        Xx = df['PRE_X'] - df['SPI_X_AVG']
        Xy = df['PRE_Y'] - df['SPI_Y_AVG']
        X =  pd.concat([Xx, Xy], axis=1).to_numpy()
        y =  df[['POST_X','POST_Y']].to_numpy()
    
    train_save_filename = f'{result_path}/regr_multirf_{args.train_data}.pkl'

    X_train, X_test, y_train, y_test = None, None, None, None
    if args.test_size == 0:
        X_train, y_train = X, y
        X_test, y_test = torch.Tensor([]), torch.Tensor([])
    else:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=args.test_size, random_state=SEED, shuffle=True)

    print('='*10,'sizes','='*10)
    print('X train:', X_train.shape,' X test:', X_test.shape)
    print('Y train:', y_train.shape,' Y test:', y_test.shape)

    ''' load RF regressor model'''
    # load from RF model
    if args.load == True:
        print('Loading models...')
        regr_multirf = joblib.load(train_save_filename)
    
    ''' train '''
    # train and save RF models
    regr_multirf = None
    if args.train:
        print('='*10,'train','='*10)
        print('Initializing Regressors...')
        regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=args.num_predictors,
                                                                max_depth=args.max_depth,
                                                                random_state=SEED))
        print('Fitting Regressors...')
        regr_multirf.fit(X_train, y_train)

        print('Saving RF Models...')
        COMPRESS_LEVEL = 3   # 3: ideal tradeoff b/w data loss and model size
        joblib.dump(regr_multirf, train_save_filename, compress=COMPRESS_LEVEL)

    ''' test '''
    y_multirf = None
    if args.test:
        print('='*10,'test','='*10)
        print('Predicting...')
        y_multirf = regr_multirf.predict(X_test)

    ''' statistics '''
    # pre_distances = [np.linalg.norm((x,y)) for x, y in zip(X_test[:,0], X_test[:, 1])]
    # post_distances = [np.linalg.norm((x,y)) for x, y in zip(y_test[:,0], y_test[:, 1])]
    # post_distances_est = [np.linalg.norm((x,y)) for x, y in zip(y_multirf[:,0], y_multirf[:, 1])]
    
    # pre_distances = pd.Series(pre_distances, dtype=float, name='pre_distances').describe()
    # post_distances = pd.Series(post_distances, dtype=float, name='post_distances').describe()
    # post_distances_est = pd.Series(post_distances_est, dtype=float, name='post_distances_est').describe()
    # stats = pd.concat([pre_distances, post_distances, post_distances_est], axis=1)
    # stats.to_csv('statistics.csv')
    
    ''' plot '''
    if args.plot:
        print('Plotting...')
        # RF multi only
        if 'y_multirf' in locals():
            plt.figure()
            s = 30
            a = 0.6
            # plt.scatter(X_test[:, 0], X_test[:, 1], edgecolor='b', c="navy", s=s, marker="s", alpha=a, label="PRE test data") # PRE
            plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='r', c="navy", s=s, marker="s", alpha=a, label="POST test data") # POST
            plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='g', c="cornflowerblue", s=s, alpha=a, label="POST estimate (Multi RF score=%.2f)" % regr_multirf.score(X_test, y_test))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend(loc='best')
            # plt.title("Random Forest (multi-output) meta estimator")
            if chip is not None:
                plt.title(f"{chip}_{args.test_size} samples_Multi-Output RF regressor") # meta estimator
                plt.savefig(f'{chip}_{args.test_size} regressor_multionly_output.png')
            else:
                plt.title(f"{args.test_size} samples_Multi-Output RF regressor")
                plt.savefig(f'regressor_multionly_output_{args.test_size}.png')
            plt.legend()
            # plt.show()
            plt.clf()
