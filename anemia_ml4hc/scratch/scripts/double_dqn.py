import pandas as pd
import numpy as np
import seaborn as sns
import random
import os
from datetime import datetime
# import torch
from stable_baselines import DQN
from stable_baselines import bench, logger
import tensorflow
import sys
sys.path.append('..')
from modules import utils, constants
from sklearn.model_selection import train_test_split
from stable_baselines.common.callbacks import CheckpointCallback


SEED = constants.SEED
random.seed(SEED)
np.random.seed(SEED)
tensorflow.set_random_seed(constants.SEED)
tensorflow.compat.v1.set_random_seed(constants.SEED)

train_df = pd.read_csv('../data/train_set_basic.csv')

X_train = train_df.iloc[:, 0:-1]
y_train = train_df.iloc[:, -1]

X_train, y_train = np.array(X_train), np.array(y_train)


if __name__ == '__main__':
    training_env = utils.create_env(X_train, y_train)
    training_env = bench.Monitor(training_env, logger.get_dir())
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, 
                learning_starts=50000, train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, 
                n_cpu_tf_sess=1, policy_kwargs=dict(dueling=False), double_q=True)


    checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='../models/sb/double_dqn', 
                                            name_prefix='double_dqn_basic')

    model.learn(total_timesteps=100000000, log_interval=500000, callback=checkpoint_callback)