
import os
import pathlib

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--max_depth', default=100, type=int, help="max-depth for the regressor")
parser.add_argument('--num_trees', default=50,type=int, help="number of estimators")
parser.add_argument('--train', default=False, action='store_true', help="flag for training")
parser.add_argument('--test', default=False, action='store_true', help='flag to test the model')
parser.add_argument('--load', default=False, action='store_true', help='flag to load a model')
parser.add_argument('--load_path', default=None, type=str, help='path to the model pickle file')
parser.add_argument('--type', default=2, type=int, help="1: gradient boosting, 2: random forest")
parser.add_argument('--test_size', default=0, type=int, help="test sample size (to plot)")
args = parser.parse_args()

''' ===== config ===== '''
train = args.train
test = plot = args.test
load = True if args.load_path is not None else False
max_depth = args.max_depth
num_trees = args.num_trees
TEST_SIZE = args.test_size         # test sample size after split; used for plotting as well
regressor_type = args.type

regressor = None
if regressor_type == 1: #'gradientboosting'
    regressor = GradientBoostingRegressor
elif regressor_type == 2: # 'randomforest'
    regressor = RandomForestRegressor
regr2name = {1: 'gradient_boosting', 2: 'random_forest'}

''' set data path'''
dir_path = pathlib.Path().absolute()
file = './data/imputed_data.csv'
data_path = os.path.join(dir_path, file)

''' set result path '''
result_path = './models'
if not os.path.isdir(result_path):
    os.makedirs(result_path, mode=0o755, exist_ok=True)

''' read data '''
df = pd.read_csv(data_path  ,index_col=False).drop(['Unnamed: 0'], axis=1)
df.reset_index(drop=True, inplace=True)
assert df.isnull().sum().sum() == 0

SEED = 42                   # for random forest regressor
np.random.RandomState(SEED) # for train-test split

# (1) all chips
chip = 'all'
# df = df[df['PartType']==chip]

# X = df[['PRE_L','PRE_W']].to_numpy()
# y = df[['POST_L','POST_W']].to_numpy()

# Xx = df['PRE_X'] - df['SPI_X_AVG']
# Xy = df['PRE_Y'] - df['SPI_Y_AVG']
# X =  pd.concat([Xx, Xy], axis=1).to_numpy()

X = df[['PRE_L','PRE_W','SPI_L','SPI_W','SPI_VOLUME_MEAN']].to_numpy()
y = df[['POST_L','POST_W']].to_numpy()
train_save_filename = f'./{result_path}/regr_multirf_{chip}_{args.num_trees}_trees_{args.max_depth}_deep_{regr2name[regressor_type]}.pkl'
# set this to final model
# train_save_filename = 'reflow_oven.pkl'

X_train, X_test, y_train, y_test = None, None, None, None
if TEST_SIZE == 0:
    X_train, y_train = X, y
    X_test, y_test = torch.Tensor([]), torch.Tensor([])
elif TEST_SIZE > 0:
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True)

print('='*10,'sizes','='*10)
print('X train:', X_train.shape,' X test:', X_test.shape)
print('Y train:', y_train.shape,' Y test:', y_test.shape)

''' how to choose the number of predictors for random forest
- if the number of observatiosn is large, but the number of trees is too small, 
then some observations will be predicted only once or even not at all
- if the number of predictors is large but the number of trees is too small, then some features can theoretically be missed in all subspaces used
- both cases (above) result in the decrease of RF predictive power

According to [Link](https://mljar.com/blog/how-many-trees-in-random-forest/),
for a dataset of 1e5 rows, 1e3 RF predictors has a strong dependency

'''

''' train and save RF models '''
regr = None
if train == True:
    print('='*10,'train','='*10)
    print('Initializing Regressors...')
    
    regr = MultiOutputRegressor(regressor(n_estimators=num_trees, max_depth=max_depth, random_state=SEED))
    print('Fitting Regressors...')
    
    X_train = X_train.reshape(len(df)-TEST_SIZE, X.shape[1])
    y_train = y_train.reshape(len(df)-TEST_SIZE, y.shape[1])
    regr.fit(X_train, y_train)

    print('Saving RF Models...')
    COMPRESS_LEVEL = 3   # 3: ideal tradeoff b/w data loss and model size
    joblib.dump(regr, train_save_filename, compress=COMPRESS_LEVEL)

''' load from model '''
if load == True:
    print('Loading models...')
    if args.load_path is not None:
        load_path = f'./models/regressor_{chip}_{regr2name[regressor_type]}.pkl'
    else:
        load_path = args.load_path
    regr = joblib.load(load_path)
    
''' test the model '''
if TEST_SIZE == 0: test = False
y_regressor = None
if test == True:
    print('='*10,'test','='*10)
    print('Predicting...')
    
    # Predict on new data
    X_test = X_test.reshape(TEST_SIZE, X.shape[1])
    y_regressor = regr.predict(X_test)

''' plot results '''
if plot == True:
    print('Plotting...')

    # RF multi only
    if 'y_regressor' in locals():
        plt.figure()
        s = 20
        a = 0.4
        plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k', c="green", s=s, marker="^", alpha=a, label="Actual")
        plt.scatter(y_regressor[:, 0], y_regressor[:, 1], edgecolor='k', c="cornflowerblue", s=s, alpha=a, marker="o",
                    label=f"regressor (score={regr.score(X_test, y_test):.2f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"chip: {chip},{regr2name[regressor_type]} regressor({args.num_trees} trees, {args.max_depth} deep) ({TEST_SIZE} samples)")
        plt.legend()

        plt.savefig(f'chip_{chip}-{regr2name[regressor_type]} regressor({args.num_trees} trees, {args.max_depth} deep)_{TEST_SIZE}_samples.png')
        print('saved regressor output')
        
        # plt.show()
        # plt.clf()


