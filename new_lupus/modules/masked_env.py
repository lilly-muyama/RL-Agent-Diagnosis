import os
import random
import copy
import numpy as np
from modules import constants #changed this
import gym
from gym import Env
from gym.spaces import Discrete, Box

random.seed(constants.SEED)
np.random.seed(constants.SEED)
os.environ['PYTHONHASHSEED']=str(constants.SEED)

class LupusEnv(Env):
    def __init__(self, X, Y, random=True):
        super(LupusEnv, self).__init__()
        self.X = X
        self.Y = Y
        self.feat_num = self.X.shape[1]
        self.actions = constants.ACTION_SPACE
        self.action_space = Discrete(len(self.actions))
        self.observation_space = Box(np.inf, np.inf, shape=(2, self.feat_num))
        self.random = random
        self.sample_num = len(X)
        self.idx =-1
        self.x = np.zeros((self.feat_num,), dtype=np.float32)
        self.bin_mask = np.zeros((self.feat_num,), dtype=np.float32)
        self.y = np.nan
        self.s = np.zeros((self.feat_num,), dtype=np.float32)
        self.state = self.concat_state(self.s, self.bin_mask)
        self.num_classes = constants.CLASS_NUM
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        self.seed()
        print('Using environment with binary mask for missing data')

    def seed(self, seed=constants.SEED):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')


    def reset(self, idx=None):
        # print('RESETTING!!!')
        if idx is not None:
            self.idx = idx
        elif self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx >= self.sample_num:
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.bin_mask = np.zeros((self.feat_num,), dtype=np.float32)
        self.s = np.zeros((self.feat_num,), dtype=np.float32)
        self.state = self.concat_state(self.s, self.bin_mask)
        self.trajectory = []
        self.episode_length = 0
        self.total_reward = 0
        return self.state
    
    def get_next_state(self, feature_idx):
        self.x = self.x.reshape(-1, self.feat_num)
        x_value = self.x[0, feature_idx]
        # print(f'x_value: {x_value}')
        next_s = copy.deepcopy(self.s)
        # print(f'first next s: {next_s}')
        next_bin_mask = copy.deepcopy(self.bin_mask)
        if x_value != -1:
            next_s[feature_idx] = x_value
            # print(f'second next s: {next_s}')
        else: # if value is missing
            next_bin_mask[feature_idx] = 1
            # print(f'next bin_mask: {next_bin_mask}')
        self.s = next_s
        self.bin_mask = next_bin_mask
        next_state = self.concat_state(next_s, next_bin_mask)
        # print(f'next state: {next_state}')
        return next_state

    def concat_state(self, s, bin_mask):
        # print(f's: {s.shape}')
        # print(f'bin mask: {bin_mask.shape}')
        # s = s.reshape(-1, self.feat_num)
        # bin_mask = bin_mask.reshape(-1, self.feat_num)
        # print(f'reshaped s: {s}')
        # print(f'reshaped bin mask: {bin_mask}')
        return np.stack([s, bin_mask])


    def step(self, action):
        if isinstance(action, np.ndarray):
            # print(action)
            action == int(action)
        self.episode_length += 1
        # print(f'Step {self.episode_length} of index {self.idx}') # to delete
        # print(f'old state: {self.state}')
        # print(f'action: {action}')
        reward = 0
            
        if action < self.num_classes: # if diagnosis action
            if action == self.y:
                reward += constants.CORRECT_DIAGNOSIS_REWARD
                self.total_reward += constants.CORRECT_DIAGNOSIS_REWARD
                is_success = True
            else:
                reward += constants.INCORRECT_DIAGNOSIS_REWARD
                self.total_reward += constants.INCORRECT_DIAGNOSIS_REWARD
                is_success = False
            terminated = False
            done = True
            y_actual = self.y 
            y_pred = int(action)
            self.trajectory.append(self.actions[action])
        
        elif self.actions[action] in self.trajectory: #repeated action
            terminated = True
            reward += constants.REPEATED_ACTION_REWARD
            self.total_reward += constants.REPEATED_ACTION_REWARD
            done = True
            y_actual = self.y 
            y_pred = constants.CLASS_DICT['Inconclusive diagnosis']
            is_success = True if y_actual == y_pred else False
            self.trajectory.append('Inconclusive diagnosis')
            
        else:
            reward += 0
            self.total_reward += 0
            terminated = False
            done = False
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
            self.state = self.get_next_state(action - self.num_classes)
            self.trajectory.append(self.actions[action])
        # print(f'new state: {self.state}')

        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward':self.total_reward, 'y_pred':y_pred, 'y_actual':y_actual, 
        'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}
        # print(f'info: {info}')
        return self.state, reward, done, info

