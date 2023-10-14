import pandas as pd
import numpy as np
import random 
import os
import sys
sys.path.append('..')
import tensorflow as tf

from modules import former_constants as constants
import argparse
# from threading import Thread
from multiprocessing import Process



def run_dqn_model(model_type, seed, steps, train_size):
    # dir_name = f'seed_{seed}_{steps}'
    dir_name = f'seed_{seed}_{steps}'
    parent_dir = f'../models/logs/{model_type}/train_set_size/{train_size}/biopsy_9'
    path = os.path.join(parent_dir, dir_name)
    os.mkdir(path)

    model = utils.stable_dueling_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_dqn_per', filename=f'dueling_dqn_per_{steps}', 
                                     per=True)
        
    return model



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Parameters for dqn model')
    parser.add_argument('-t', '--steps', help='Number of timesteps to train the model', type=int, default=int(10e7))
    parser.add_argument('-s', '--seed', help='Seed for dataset to use in experiment', type=int) #added
    parser.add_argument('-z', '--size', help='Fraction of size of train set', type=float) #added
    args = parser.parse_args()
    # constants.init(args)

    from modules import utils

    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED']=str(42)
    tf.set_random_seed(42)
    tf.compat.v1.set_random_seed(42)

    print(f'Seed being used in environment: {constants.SEED}')
    print(f'Seed being used for dataset: {args.seed}')
    
    print(f'Number of steps: {args.steps}')

    train_df = pd.read_csv(f'../new_data/train_set_basic_{args.size}_seed_{args.seed}.csv')
    train_df = train_df.fillna(-1)

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]
    X_train, y_train = np.array(X_train), np.array(y_train)   


   
    model_name = 'dueling_dqn_per'

    run_dqn_model(model_name, args.seed, args.steps, args.size)   
    


    
   