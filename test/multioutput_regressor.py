# # mount your current Google Drive (Directory)
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import joblib

pd.set_option('display.max_columns', None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config
train = True
load = not train
plot = False
max_depth = 30 # 30
num_predictors = int(100)
TEST_SIZE = 100

dir_path = pathlib.Path().absolute()
print(dir_path)
file = '../data/imputed_data.csv'
data_path = os.path.join(dir_path, file)

# read data
df = pd.read_csv(data_path  ,index_col=False).drop(['Unnamed: 0'], axis=1)
df.reset_index(drop=True, inplace=True)
assert df.isnull().sum().sum() == 0

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# Create a random dataset
rng = np.random.RandomState(1)
X = df[['PRE_X','PRE_Y']].to_numpy()
y = df[['POST_X','POST_Y']].to_numpy()

# print('X:', X.shape, X[0:10], '\n','y:', y.shape, y[0:10])

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=TEST_SIZE, random_state=42, shuffle=True)

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

''' save RF models '''
regr_multirf = None
regr_rf = None
if train == True:
    print('Fitting MultiOutputRegressor...')
    # random forest regressor - multiple output
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=num_predictors,
                                                            max_depth=max_depth,
                                                            random_state=0))
    regr_multirf.fit(X_train, y_train)

    print('Fitting RandomForestRegressor...')
    # random forest regressor
    regr_rf = RandomForestRegressor(n_estimators=num_predictors, max_depth=max_depth,
                                    random_state=2)
    regr_rf.fit(X_train, y_train)

    joblib.dump(regr_multirf, "./regr_multirf.pkl", compress=3)
    joblib.dump(regr_rf, "./regr_rf.pkl", compress=3)
if load == True:
    regr_multirf = joblib.load("./regr_multirf.pkl")
    regr_rf = joblib.load("./regr_rf.pkl")

print('Predicting...')
# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

# print('x train:', X_train, '\ny train:',y_train, '\n\nx_test:',X_test,'\ny_test:',y_test,'\n\nmultiRF:',y_multirf, '\nRF:', y_rf)

if plot == True:
    # RF only
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

    plt.savefig('./regressor_only_output.png')
    plt.show()
    plt.clf()

    # RF multi only
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

    plt.savefig('./regressor_multionly_output.png')
    plt.show()
    # plt.clf()



    print('Plotting...')
    # Plot the results
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

    plt.savefig('./regressor_output.png')
    plt.show()
    # plt.clf()

