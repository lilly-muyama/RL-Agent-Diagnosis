import os
import random
import copy
import numpy as np
from modules import simple_constants, utils
from gym import Env
from gym.spaces import Discrete, Box


random.seed(simple_constants.SEED)
np.random.seed(simple_constants.SEED)
os.environ['PYTHONHASHSEED']=str(simple_constants.SEED)


class SimpleEnv(Env):
    def __init__(self, X, Y, random=True):
        super(SimpleEnv, self).__init__()
        self.action_space = Discrete(simple_constants.ACTION_NUM)
        self.observation_space = Box(0, 1.5, (simple_constants.FEATURE_NUM,))
        self.X = X
        self.Y = Y
        self.random = random
        self.sample_num = len(X)
        self.idx = -1
        self.x = np.zeros((simple_constants.FEATURE_NUM,), dtype=np.float32) 
        self.y = np.nan
        self.state = np.full((simple_constants.FEATURE_NUM,), -1, dtype=np.float32)
        self.actions = simple_constants.ACTION_SPACE
        self.num_classes = simple_constants.CLASS_NUM
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)
        self.episode_length += 1
        reward = 0
        if action < self.num_classes:
            if action == self.y:
                reward += 1
                self.total_reward += 1
                # reward += 44
                # self.total_reward += 44
                is_success=True
            else:
                reward -= 1
                self.total_reward -= 1
                # reward -= 22
                # self.total_reward -= 22
                is_success = False
            terminated = False
            done=True 
            y_actual = self.y
            y_pred = int(action)
            self.trajectory.append(self.actions[action])
        elif self.actions[action] in self.trajectory:
            terminated = True
            reward -= 1
            self.total_reward -= 1
            terminated = True
            done=True
            y_actual = self.y
            y_pred = simple_constants.CLASS_DICT['Inconclusive diagnosis']
            is_success = True if y_actual == y_pred else False
            self.trajectory.append('Inconclusive diagnosis')
        else:
            # if self.actions[action] == 'ana':
            #     reward += 1
            #     self.total_reward+=1
            # else:
            reward -= 0
            self.total_reward -= 0
            terminated = False
            done = False
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
            self.state = self.get_next_state(action - self.num_classes)
            # if x_value == 1:
            #     reward -= 0
            #     self.total_reward -= 0
            # else:
            #     reward -= 1
            #     self.total_reward -= 1

            self.trajectory.append(self.actions[action])

        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward':self.total_reward, 'y_pred':y_pred, 'y_actual':y_actual, 
        'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}
        # print(f'info: {info}')
        return self.state, reward, done, info
    
    def get_next_state(self, feature_idx):
        self.x = self.x.reshape(-1, simple_constants.FEATURE_NUM)
        x_value = self.x[0, feature_idx]
        next_state = copy.deepcopy(self.state)
        next_state[feature_idx] = x_value
        return next_state
    
    def reset(self, idx=None):
        if idx is not None:
            self.idx = idx
        elif self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx == self.sample_num:
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.state = np.full((simple_constants.FEATURE_NUM,), -1, dtype=np.float32)
        self.trajectory = []
        self.episode_length = 0
        self.total_reward = 0
        return self.state

    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')


    
    


    


