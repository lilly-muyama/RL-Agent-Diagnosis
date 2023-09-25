import pandas as pd
import numpy as np
import random 
import os
import sys
sys.path.append('..')
import tensorflow as tf
from sklearn.impute import KNNImputer
from modules.constants import constants
import argparse



def run_dqn_model(model_type, seed, steps):
    # dir_name = f'seed_{seed}_{steps}'
    dir_name = f'seed_{seed}_{steps}'
    parent_dir = f'../models/logs/{model_type}/missingness/0.1/biopsy_9/knn_imputer/default_mean_k_1'
    path = os.path.join(parent_dir, dir_name)
    os.mkdir(path)
  
    if model_type=='dqn':
        model = utils.stable_vanilla_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dqn', filename=f'dqn_{steps}')
    elif model_type=='ddqn':
        model = utils.stable_double_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='ddqn', filename=f'ddqn_{steps}')
    elif model_type== 'dueling_dqn':
        model = utils.stable_dueling_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_dqn', filename=f'dueling_dqn_{steps}')
    elif model_type == 'dueling_ddqn':
        model = utils.stable_dueling_ddqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_ddqn', filename=f'dueling_ddqn_{steps}')
    elif model_type =='dqn_per':
        model = utils.stable_vanilla_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dqn_per', filename=f'dqn_per_{steps}', per=True)
    elif model_type == 'ddqn_per':
        model = utils.stable_double_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='ddqn_per', filename=f'ddqn_per_{steps}', per=True)
    elif model_type == 'dueling_dqn_per':
        model = utils.stable_dueling_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_dqn_per', filename=f'dueling_dqn_per_{steps}', per=True)
    elif model_type == 'dueling_ddqn_per':
        model = utils.stable_dueling_ddqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_ddqn_per', filename=f'dueling_ddqn_per_{steps}', per=True)
    else:
        raise ValueError('Unknown model type!')
    return model



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Parameters for dqn model')
    parser.add_argument('-s', '--seed', help='Seed to use in experiment', type=int) #added
    args = parser.parse_args()
    constants.init(args)

    if args.seed:
        print(f'Using seed: {args.seed}')
    else:
        print('No seed provided')

    from modules import utils

    random.seed(constants.SEED)
    np.random.seed(constants.SEED)
    os.environ['PYTHONHASHSEED']=str(constants.SEED)
    tf.set_random_seed(constants.SEED)
    tf.compat.v1.set_random_seed(constants.SEED)

    print(f'Seed being used: {constants.SEED}')

    train_df = pd.read_csv('../new_data/knn_imputed/default_mean_k_1/missingness_0.1.csv')

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]
    X_train, y_train = np.array(X_train), np.array(y_train)   


    steps = int(100e6)
    model_name = 'dueling_ddqn_per'

    run_dqn_model(model_name, constants.SEED, steps)
    
   
