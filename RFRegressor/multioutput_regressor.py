# mount your current Google Drive (Directory)
import os
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import torch

pd.set_option('display.max_columns', None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config
train = True
load = not train
plot = False
max_depth = 30
num_predictors = 100    # needs to be INTEGER
TEST_SIZE = 100         # test sample size after split

# data path
dir_path = pathlib.Path().absolute()
file = '../data/imputed_data.csv'
data_path = os.path.join(dir_path, file)

''' path '''
# result path
result_path = './models'
if not os.path.isdir(result_path):
    os.makedirs(result_path, mode=0o755, exist_ok=True)

# read data
df = pd.read_csv(data_path  ,index_col=False).drop(['Unnamed: 0'], axis=1)
df.reset_index(drop=True, inplace=True)
assert df.isnull().sum().sum() == 0

''' define dataset '''
# Create a random dataset
SEED = np.random.RandomState(1)
X = df[['PRE_X','PRE_Y']].to_numpy()
y = df[['POST_X','POST_Y']].to_numpy()

# print('X:', X.shape, X[0:10], '\n','y:', y.shape, y[0:10])

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
regr_multirf = None
regr_rf = None
if train == True:
    print('='*10,'train','='*10)
    print('Fitting MultiOutputRegressor...')
    # random forest regressor - multiple output
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=num_predictors,
                                                            max_depth=max_depth,
                                                            random_state=0))
    # random forest regressor
    # regr_rf = RandomForestRegressor(n_estimators=num_predictors, max_depth=max_depth,
    #                                 random_state=2)

    print('Fitting RandomForestRegressor...')
    regr_multirf.fit(X_train, y_train)
    # regr_rf.fit(X_train, y_train)

    print('Saving RF Models...')
    COMPRESS_LEVEL = 3   # 3: ideal tradeoff b/w data loss and model size
    joblib.dump(regr_multirf, f'./{result_path}/regr_multirf.pkl', compress=COMPRESS_LEVEL)
    # joblib.dump(regr_rf, f'./{result_path}/regr_rf.pkl', compress=COMPRESS_LEVEL)

''' load from RF model '''
if load == True:
    print('Loading models...')
    regr_multirf = joblib.load(f'./{result_path}/regr_multirf.pkl')
    # regr_rf = joblib.load("./regr_rf.pkl")

print('='*10,'test','='*10)
print('Predicting...')
# Predict on new data
y_multirf = regr_multirf.predict(X_test)
# y_rf = regr_rf.predict(X_test)

# print('x train:', X_train, '\ny train:',y_train, '\n\nx_test:',X_test,'\ny_test:',y_test,'\n\nmultiRF:',y_multirf, '\nRF:', y_rf)

if plot == True:
    print('Plotting...')
    
    # RF only
    if 'y_rf' in locals():
        plt.figure()
        s = 50
        a = 0.4
        plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
                    c="navy", s=s, marker="s", alpha=a, label="Data")
        plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
                    c="c", s=s, marker="^", alpha=a,
                    label="RF score=%.2f" % regr_rf.score(X_test, y_test))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Random Forest (multi-output) meta estimator")
        plt.legend()

        plt.savefig(f'./regressor_only_output.png')
        # plt.show()
        # plt.clf()
        print('saved RF')
    
    # RF multi only

    if 'y_multirf' in locals():
        plt.figure()
        s = 50
        a = 0.4
        plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
                    c="navy", s=s, marker="s", alpha=a, label="Data")
        plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
                    c="cornflowerblue", s=s, alpha=a,
                    label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Random Forest (multi-output) meta estimator")
        plt.legend()

        plt.savefig(f'./regressor_multionly_output.png')
        # plt.show()
        # plt.clf()
        print('saved multi RF')

    # RF and Multi RF together
    if 'y_rf' in locals() and 'y_multirf' in locals():
        plt.figure()
        s = 50
        a = 0.4
        plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
                    c="navy", s=s, marker="s", alpha=a, label="Data")
        plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
                    c="cornflowerblue", s=s, alpha=a,
                    label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
        plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
                    c="c", s=s, marker="^", alpha=a,
                    label="RF score=%.2f" % regr_rf.score(X_test, y_test))
        # plt.xlim([-6, 6])
        # plt.ylim([-6, 6])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Random Forest (multi-output) meta estimator")
        plt.legend()

        plt.savefig(f'./regressor_both_output.png')
        # plt.show()
        # plt.clf()
        print('saved both')

