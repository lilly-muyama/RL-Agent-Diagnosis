import os
import random
import copy
import numpy as np
from modules import constants, utils
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
        # print(f'Step {self.episode_length+1} for index {self.idx}')
        # print(f'Current state: {self.state}')
        if isinstance(action, np.ndarray):
            #print('CONVERTING ACTION')
            action == int(action)
        # print(f'action: {action}')
        self.episode_length += 1
        reward = 0
        if (self.episode_length == self.max_steps) & (action >= self.num_classes):
            # print(f'maximum steps reached for index: {self.idx}')
            # reward -= 1
            # self.total_reward -=1
            reward -= 1
            self.total_reward -= 1
            terminated = True
            done = True
            y_actual = self.y 
            y_pred = int(constants.CLASS_DICT['Inconclusive diagnosis'])
            # self.trajectory.append(self.actions[action])
            self.trajectory.append('Inconclusive diagnosis')
            self.state = self.get_next_state(action - self.num_classes)
            is_success = True if y_actual == y_pred else False
            
        elif action < self.num_classes: # if diagnosis action
            if action == self.y:
                # print(f'correct diagnosis action for index: {self.idx}')
                # reward += 1
                # self.total_reward += 1
                reward += 1
                self.total_reward += 1
                is_success = True
            else:
                # print(f'incorrect diagnosis action for index: {self.idx}')
                # reward -= 1
                # self.total_reward -= 1
                reward -= 1
                self.total_reward -= 1
                is_success = False
            terminated = False
            done = True
            y_actual = self.y 
            y_pred = int(action)
            self.trajectory.append(self.actions[action])
        elif self.actions[action] in self.trajectory: #repeated action
            # print(f'repeated action for index: {self.idx}')
            action = constants.CLASS_DICT['Inconclusive diagnosis']
            terminated = True
            # reward -= 1
            # self.total_reward -= 1
            reward -= 1
            self.total_reward -= 1
            done = True
            y_actual = self.y 
            y_pred = int(action)
            is_success = True if y_actual == y_pred else False
            self.trajectory.append(self.actions[action])
        else:
            # print(f'normal action for index: {self.idx}')
            terminated = False
            # reward -= 0
            # self.total_reward -= 0
            # if action == self.num_classes:
            reward -= 0
            self.total_reward -= 0
            # else:
            # reward -= 1
            # self.total_reward -= 1
            done = False
            self.state = self.get_next_state(action - self.num_classes)
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
            self.trajectory.append(self.actions[action])
        # print(f'Next state: {self.state}')
        
        episode_score = utils.compute_score(self.state) if done else np.nan
        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward':self.total_reward, 'y_pred':y_pred, 'y_actual':y_actual, 
        'trajectory':self.trajectory, 'terminated':terminated, 'score':episode_score,'is_success': is_success}
        # print(f'info: {info}')
        return self.state, reward, done, info


    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')

    
    def reset(self, idx=None):
        # print('RESETTING ENVIRONMENT!!!!!!!!!!!!!!!!!!!!!')
        if idx is not None:
            self.idx = idx
        elif self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx == len(self.X):
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        # print(f'self.idx: {self.idx}')
        # print(f'self.x: {self.x}')
        # print(f'self.y: {self.y}')
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




