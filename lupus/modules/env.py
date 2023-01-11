import os
import random
import numpy as np
from modules import constants
from gym import Env
from gym.spaces import Discrete, Box


random.seed(constants.SEED)
np.random.seed(constants.SEED)
os.environ['PYTHONHASHSEED']=str(constants.SEED)


class LupusEnv(Env):
    def __init__(self, X, Y, random=True):
        super(LupusEnv, self).__init__()
        self.action_space = Discrete(constants.ACTION_NUM)
        self.observation_space = Box(constants.GYM_BOX_LOW, constants.GYM_BOX_HIGH, (constants.FEATURE_NUM,))
        self.actions = constants.ACTION_SPACE
        self.max_steps = constants.MAX_STEPS
        self.X = X
        self.Y = Y
        self.sample_num = len(X)
        self.idx =-1
        self.x = np.zeros((constants.FEATURE_NUM,), dtype=np.float32)
        self.y = np.nan
        self.state = np.full((constants.FEATURE_NUM,), -1, dtype=np.float32)
        self.num_classes = constants.CLASS_NUM
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        self.random = random

    def step(self, action):
        if isinstance(action, np.ndarray):
            action == int(action)
        self.episode_length += 1
        reward = 0
        if self.episode_length == self.max_steps:
            reward -= 1
            self.total_reward -=1
            terminated = True
            done = True
            y_actual = self.y 
            y_pred = constants.CLASS_DICT['Inconclusive diagnosis']
            is_success = True if y_actual == y_pred else False
        elif action < self.num_classes: # if diagnosis action
            if action == self.y:
                reward += 1
                self.total_reward += 1
                is_success = True
            else:
                reward -= 1
                self.total_reward -= 1
                is_success = False
            terminated = False
            done = True
            y_actual = self.y 
            y_pred = action
        elif self.actions[action] in self.trajectory: #repeated action
            action = constants.CLASS_DICT['Inconclusive diagnosis']
            terminated = True
            reward -= 1
            self.total_reward -= 1
            done = True
            y_actual = self.y 
            y_pred == action
            is_success = True if y_actual == y_pred else False
        else:
            terminated = False
            reward += 0
            self.total_reward += 0
            done = False
            self.state = self.get_next_state(action - self.num_classes)
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        self.trajectory.append(self.actions[action])
        episode_score = utils.compute_score(self.state) if done else np.nan
        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward':self.total_reward, 'y_pred':y_pred, 'y_actual':y_actual, 
        'trajectory':self.trajectory, 'terminated':terminated, 'score':episode_score,'is_success': is_success}
        return self.state, reward, done, info


    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')

    
    def reset(self, idx=None):
        if idx is not None:
            self.idx = idx
        elif self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx == len(self.X):
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.state = np.full((constants.FEATURE_NUM,), -1, dtype=np.float32)
        self.trajectory = []
        self.episode_length = 0
        self.total_reward = 0
        return self.state

        
    def get_next_state(self, feature_idx):
        self.x = self.x.reshape(-1, constants.FEATURE_NUM)
        x_value = self.x[0, feature_idx]
        next_state = copy.deepcopy(self.state)
        next_state[feature_idx] = x_value
        return next_state




