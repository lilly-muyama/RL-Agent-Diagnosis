import pandas as pd
import numpy as np
import random 
import os
import sys
sys.path.append('..')
import tensorflow as tf
from modules import utils, constants
import argparse

SEED = constants.SEED
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
tf.set_random_seed(constants.SEED)
tf.compat.v1.set_random_seed(constants.SEED)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Parameters for ppo model')
    parser.add_argument('-t', '--steps', help='Number of timesteps to train the model', required=True, type=int)
    parser.add_argument('-s', '--seed', help='Seed to use in experiment', type=int, default=42) #added
    args = parser.parse_args()

    
    train_df = pd.read_csv(f'../data/train_set_basic.csv')
    train_df = train_df.fillna(-1)

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]

    X_train, y_train = np.array(X_train), np.array(y_train)

    model_name =f'ppo_basic_{SEED}_{args.steps}' 
    ppo_model = utils.stable_ppo(X_train, y_train, args.steps, True, f'../models/{model_name}')
    

    print('Training complete and model saved ...')
    