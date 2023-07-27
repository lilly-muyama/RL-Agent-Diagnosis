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

if __name__=='__main__':
    pass