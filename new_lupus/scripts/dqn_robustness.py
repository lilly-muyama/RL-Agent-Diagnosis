import pandas as pd
import numpy as np
import random 
import os
import sys
sys.path.append('..')


from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
from torch.nn import functional as F

from modules.constants import constants
import argparse
# from threading import Thread
from multiprocessing import Process

def create_env(X, y, random=True):
    '''
    Creates and environment using the given data
    '''
    env = LupusEnv(X, y, random)
    print(f'The environment seed is {env.seed()}') #to delete
    return env

def stable_baselines3_robust_dqn(X_train, y_train, steps, save, log_path, log_prefix, filename, beta, al_r, al_p, 
                                 p_proxy):
    training_env = create_env(X_train, y_train)
    model = RobustDQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, beta=beta, al_r=al_r, al_p=al_p,
                     p_proxy=p_proxy)
    checkpoint_callback = CheckpointCallback(save_freq=constants.CHECKPOINT_FREQ, save_path=log_path, 
                                            name_prefix=log_prefix)
    model.learn(total_timesteps=steps, log_interval=100000, callback=checkpoint_callback)
    if save:
        model.save(f'{log_path}/{filename}_full_model')
    training_env.close()
    return model

def run_robust_dqn_model(X_train, y_train, steps, seed, beta, al_r, al_p, p_proxy, parent_dir):
    #create directory where to store the DQN checkpoints
    dir_name = f'seed_{seed}_{steps}'
    path = os.path.join(parent_dir, dir_name)
#     os.mkdir(path)
    model = stable_baselines3_robust_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='robust_dqn3', 
                                  filename=f'robust_dqn3_{steps}', beta=beta, al_r=al_r, al_p=al_p, p_proxy=p_proxy)
    return model



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Parameters for dqn model')
    parser.add_argument('-t', '--steps', help='Number of timesteps to train the model', type=int, default=int(10e7))
    parser.add_argument('-s', '--seed', help='Seed to use in experiment', type=int, default=42) #added
    parser.add_argument('-m', '--noisiness', help='Level of noise in the dataset', type=float) #added
    parser.add_argument('-b', '--beta', help = 'Beta', type=int, default=1)
    parser.add_argument('-r', '--al_r', help = 'al_r', type=float, default=0.01)
    parser.add_argument('-p', '--al_p', help = 'al_p', type=float, default=0.01)
    parser.add_argument('-q', '--p_proxy', help = 'p_proxy', type=str, default='l2-norm')
    parser.add_argument('-l', '--reward_beta', help = 'beta used in reward function', default=9)
    args = parser.parse_args()
    constants.init(args)

    from modules.scripts_env import LupusEnv
    from modules.robust_dqn import RobustDQN

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED']=str(args.seed)
    th.manual_seed(args.seed)
    th.use_deterministic_algorithms(True)
   

    print(f'Seed being used: {constants.SEED}')
    print(f'Number of steps: {args.steps}')
    print(f'Beta being used: {constants.BETA}')

    train_df = pd.read_csv(f'../new_data/train_set_noisiness_{args.noisiness}.csv')
    # train_df = train_df.fillna(-1)

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]
    X_train, y_train = np.array(X_train), np.array(y_train)   

    parent_dir = f'../models/logs/robust_dqn3/trial'
    run_robust_dqn_model(X_train, y_train, args.steps, args.seed, args.beta, args.al_r, args.al_p, args.p_proxy, parent_dir)
    

    
   