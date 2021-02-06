
import argparse
import os
import sys
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

'''
get MOM4 data by chip
'''
def customMOM4chipsample(df: pd.DataFrame, input_vars: list, num_samples: int, chiptype: str = None, random_state: int = 42):
    
    # select the dataframe for the chip type in conifg
    chip_df = None
    if chiptype == None:
        chip_df = df
    else:
        for name, group in df.groupby(['PartType']):
            if name == chiptype:
                chip_df = group
            else:
                continue
    # if none, there is no value for that chip
    assert chip_df is not None, '[Error] check chip type' 
    
    sampled_chips = chip_df.sample(n=num_samples, random_state=random_state)[input_vars]
    return sampled_chips

def parse_args():
    ''' how to choose the number of predictors for random forest
    - if the number of observatiosn is large, but the number of trees is too small, 
    then some observations will be predicted only once or even not at all
    - if the number of predictors is large but the number of trees is too small, then some features can theoretically be missed in all subspaces used
    - both cases (above) result in the decrease of RF predictive power

    According to [Link](https://mljar.com/blog/how-many-trees-in-random-forest/),
    for a dataset of 1e5 rows, 1e3 RF predictors has a strong dependency
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', default=100, type=int, help="max-depth for the regressor")
    parser.add_argument('--num_trees', default=50,type=int, help="number of estimators")
    parser.add_argument('--train', default=False, action='store_true', help="flag for training")
    parser.add_argument('--test', default=False, action='store_true', help='flag to test and plot the model')
    parser.add_argument('--load_path', default=None, type=str, help='path to the model pickle file')
    parser.add_argument('--type', default=2, type=int, help="1: gradient boosting, 2: random forest")
    parser.add_argument('--test_size', default=200, type=int, help="test sample size (to plot)")
    parser.add_argument('--chip', default='R1005', type=str, help="chip to sample test data from")
    args = parser.parse_args()
    return args

def create_self_alignment_model(args):
    ''' ===== config ===== '''
    if args.test == False:
        args.test_size = 0
    load = True if args.load_path is not None else False
    max_depth = args.max_depth
    num_trees = args.num_trees
    regressor_type = args.type

    regressor = None
    if regressor_type == 1: #'gradientboosting'
        regressor = GradientBoostingRegressor
    elif regressor_type == 2: # 'randomforest'
        regressor = RandomForestRegressor
    regr2name = {1: 'gradient_boosting', 2: 'random_forest'}

    ''' set data path'''
    dir_path = pathlib.Path().absolute()
    file = '../data/imputed_data.csv'
    data_path = dir_path / file

    ''' read data '''
    df = pd.read_csv(data_path  ,index_col=False).drop(['Unnamed: 0'], axis=1)
    df.reset_index(drop=True, inplace=True)
    assert df.isnull().sum().sum() == 0, 'there is a NULL value in the loaded data'

    SEED = 42                   # for random forest regressor
    np.random.RandomState(SEED) # for train-test split
    regr = None

    if args.train:
        # (1) all chips
        # chip = 'all'
        chip = args.chip
        df = df[df['PartType']==chip] if chip is not 'all' else df
        # X =  pd.concat([df['PRE_X'] - df['SPI_X_AVG'], df['PRE_Y'] - df['SPI_Y_AVG']], axis=1).to_numpy()
        X = df[['PRE_L','PRE_W','SPI_L','SPI_W','SPI_VOLUME_MEAN']].to_numpy()
        y = df[['POST_L','POST_W']].to_numpy()

        ''' set result path '''
        result_path = './models'
        if not os.path.isdir(result_path):
            os.makedirs(result_path, mode=0o755, exist_ok=True)
        train_save_filename = f'{result_path}/regressor_{chip}_{args.num_trees}_trees_{args.max_depth}_deep_{regr2name[regressor_type]}.pkl'
        # set this to final model
        # train_save_filename = 'reflow_oven.pkl'

        X_train, X_test, y_train, y_test = None, None, None, None
        if args.test_size == 0:
            X_train, y_train = X, y
            X_test, y_test = torch.Tensor([]), torch.Tensor([])
        elif args.test_size > 0:
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=args.test_size, random_state=SEED, shuffle=True)
        print('='*10,'sizes','='*10)
        print('X train:', X_train.shape,' X test:', X_test.shape)
        print('Y train:', y_train.shape,' Y test:', y_test.shape)
        print(f'[INFO] model name set as: "{train_save_filename}"')

        ''' train and save RF models '''
        print('='*10,'train','='*10)
        print('Initializing Regressors...')
        
        regr = MultiOutputRegressor(regressor(n_estimators=num_trees, max_depth=max_depth, random_state=SEED))
        print('Fitting Regressors...')
        
        X_train = X_train.reshape(len(df)-args.test_size, X.shape[1])
        y_train = y_train.reshape(len(df)-args.test_size, y.shape[1])
        regr.fit(X_train, y_train)

        print('Saving RF Models...')
        COMPRESS_LEVEL = 3   # 3: ideal tradeoff b/w data loss and model size
        joblib.dump(regr, train_save_filename, compress=COMPRESS_LEVEL)

    ''' load from model '''
    if load == True:
        print('Loading models...')
        if args.load_path is None:
            load_path = f'./models/regressor_{chip}_{regr2name[regressor_type]}.pkl'
        else:
            load_path = args.load_path
        regr = joblib.load(load_path)
        
    ''' test the model '''
    if args.test_size == 0: test = False
    y_regressor = None
    if args.test == True:
        print('='*10,'test','='*10)
        print('Predicting...')
        
        # test with a few samples
        chip = args.chip
        input_vars = ['PRE_L','PRE_W','SPI_L','SPI_W','SPI_VOLUME_MEAN']
        X_test = customMOM4chipsample(df, input_vars=input_vars, num_samples=args.test_size, chiptype=chip, random_state=SEED)
        print('='*10, f'using {args.test_size} samples', '='*10)
        print('chip: {chip}, input variables: {input_vars}')
        print('X test:', X_test.shape)

        # Predict on new data
        y_regressor = pd.DataFrame(regr.predict(X_test))

        # make into dataframe to organize better
        X_test.reset_index(drop=True, inplace=True)
        y_regressor.reset_index(drop=True, inplace=True)
        assert len(X_test) == len(y_regressor)

        ''' plot results '''
        print('Plotting...')

        # RF multi only
        if 'y_regressor' in locals():
            # plt.figure()
            fig, ax = plt.subplots(figsize=(8, 6))
            s = 100
            a = 0.4
            # plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k', c="green", s=s, marker="^", alpha=a, label="Actual")

            # PRE
            for idx, row in X_test.iterrows():
                ax.scatter(row[0], row[1], edgecolor='g', c="yellow", s=s, alpha=a, marker="o", label=f'PRE')
                # add text
                ax.annotate(f'({idx})', xy=(row[0], row[1])) #, xytext=(row[0]+1, row[1]+1))
            
            # POST
            for idx, row in y_regressor.iterrows():
                plt.scatter(row[0], row[1], edgecolor='k', c="cornflowerblue", s=s, alpha=a, marker="o",
                            label=f'POST')
                            # label=f"regressor (score={regr.score(X_test, y_test):.2f})")
                # add text
                ax.annotate(f'({idx})', xy=(row[0], row[1])) #, xytext=(row[0]+1, row[1]+1))

            # summary
            for (i1, r1), (i2, r2) in zip(X_test.iterrows(), y_regressor.iterrows()):
                assert i1 == i2
                info = f'({i1}) PRE: {np.linalg.norm((r1[0], r1[1])):.1f}, SPI: {np.linalg.norm((r1[2], r1[3])):.1f}, VOL: {r1[4]:.1f} -> POST: {np.linalg.norm((r2[0], r2[1])):.1f}'
                ax.text(0, 240+(i1+i2)*-5, info, fontsize=8)

            ax.set_xlabel("L")
            ax.set_ylabel("W")
            ax.set_title(f"chip: {chip},{regr2name[regressor_type]} regressor({args.num_trees} trees, {args.max_depth} deep) ({args.test_size} samples)")
            if args.chip == 'R0402':
                ax.set_ylim([-100, 100])
                ax.set_xlim([-150, 150])    
            elif args.chip == 'R0603':
                ax.set_ylim([-100, 100])
                ax.set_xlim([-150, 150])
            else:
                ax.set_ylim([-250, 250])
                ax.set_xlim([-250, 250])
            ax.grid(True)
            # plt.legend()

            plt.savefig(f'chip_{chip}-{regr2name[regressor_type]} regressor({args.num_trees} trees, {args.max_depth} deep)_{args.test_size}_samples.png')
            print(f'saved regressor output to: chip_{chip}-{regr2name[regressor_type]} regressor({args.num_trees} trees, {args.max_depth} deep)_{args.test_size}_samples.png')
            
            # plt.show()
            # plt.clf()


if __name__ == '__main__':
    create_self_alignment_model(parse_args())
