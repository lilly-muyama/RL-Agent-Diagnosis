import os
import copy
import random
import numpy as np
import pandas as pd
from gym import Env
#import tensorflow
from gym.spaces import Discrete, Box
import constants

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
#tensorflow.set_random_seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)


class SyntheticSimpleEnv(Env):
    def __init__(self, X, Y, random=True):
        super(SyntheticSimpleEnv, self).__init__()
        self.action_space = Discrete(6)
        self.observation_space = Box(0, 1.5, (3,))
        self.actions = ['A', 'B', 'C', 'length', 'width', 'height']
        self.max_steps = 7
        self.X = X
        self.Y = Y
        self.sample_num = len(X)
        self.idx = -1
        self.x = np.zeros((3,), dtype=np.float32)
        self.y = np.nan
        self.state = np.zeros((3,), dtype=np.float32)
        self.num_classes = 3
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        self.random = random
        
    def step(self, action):
        self.episode_length += 1
        reward = 0
        if self.episode_length == self.max_steps: 
            reward -=1
            self.total_reward -=1
            terminated = True
            done = True
            y_actual = self.y
            y_pred = np.nan
            is_success = False
        elif action < self.num_classes: 
            if action == self.y:
                reward +=1
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
        elif self.actions[action] in self.trajectory: 
            terminated = False
            reward -= 1
            self.total_reward -= 1
            done = False
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        else: 
            terminated = False
            reward += 1
            self.total_reward += 1
            done = False
            self.state = self.get_next_state(action-self.num_classes)
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        self.trajectory.append(self.actions[action])
        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward': self.total_reward, 'y_pred': y_pred, 
                'y_actual': y_actual, 'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}
        return self.state, reward, done, info         
    
    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'x: {self.x}')
        print(f'y: {self.y}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')  
    
    def reset(self):
        if self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx == len(self.X):
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.state = np.zeros((3,), dtype=np.float32)
        self.trajectory = []
        self.episode_length = 0
        self.total_reward = 0
        return self.state
    
    def get_next_state(self, feature_idx):
        self.x = self.x.reshape(-1, 3)
        x_value = self.x[0, feature_idx]
        next_state = copy.deepcopy(self.state)
        next_state[feature_idx] = x_value
        return next_state

class SyntheticComplexEnv(Env):
    def __init__(self, X, Y, random=True):
        super(SyntheticComplexEnv, self).__init__()
        self.action_space = Discrete(14)
        self.observation_space = Box(0, 1.5, (8,))
        self.actions = ['Hemolytic anemia', 'Anemia of chronic disease', 'Aplastic anemia', 'Iron deficiency anemia', 
        'Vitamin B12/Folate deficiency anemia', 'Thalassemia', 'ferritin', 'ret_count', 'segmented_neutrophils', 'iron', 'tibc', 'rbc',
        'mcv', 'mentzer_index']
        self.max_steps = 10
        self.X = X
        self.Y = Y
        self.sample_num = len(X)
        self.idx = -1
        self.x = np.zeros((8,), dtype=np.float32)
        self.y = np.nan
        self.state = np.zeros((8,), dtype=np.float32)
        self.num_classes = 6
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        self.random = random
        
    
    def step(self, action):
        self.episode_length += 1
        reward = 0
        if self.episode_length == self.max_steps: 
            reward -=1
            self.total_reward -=1
            terminated = True
            done = True
            y_actual = self.y
            y_pred = np.nan
            is_success = False
        elif action < self.num_classes: 
            if action == self.y:
                reward +=1
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
        elif self.actions[action] in self.trajectory: 
            terminated = False
            reward -= 1
            self.total_reward -= 1
            done = False
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        else: 
            terminated = False
            reward += 1
            self.total_reward += 1
            done = False
            self.state = self.get_next_state(action-self.num_classes)
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        self.trajectory.append(self.actions[action])
        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward': self.total_reward, 'y_pred': y_pred, 
                'y_actual': y_actual, 'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}
        return self.state, reward, done, info
            
    
    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')
        
            
    
    def reset(self):
        if self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx == len(self.X):
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.state = np.zeros((8,), dtype=np.float32)
        self.trajectory = []
        self.episode_length = 0
        self.total_reward = 0
        return self.state
        
    
    def get_next_state(self, feature_idx):
        self.x = self.x.reshape(-1, 8)
        x_value = self.x[0, feature_idx]
        next_state = copy.deepcopy(self.state)
        next_state[feature_idx] = x_value
        return next_state


class SyntheticComplexHbEnv(Env):
    # The environment to use !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, X, Y, random=True):
        super(SyntheticComplexHbEnv, self).__init__()
        self.action_space = Discrete(constants.ACTION_NUM)
        self.observation_space = Box(0, 1.5, (constants.FEATURE_NUM,))
        self.actions = constants.ACTION_SPACE
        self.max_steps = constants.MAX_STEPS
        self.X = X
        self.Y = Y
        self.sample_num = len(X)
        self.idx = -1
        self.x = np.zeros((constants.FEATURE_NUM,), dtype=np.float32)
        self.y = np.nan
        self.state = np.zeros((constants.FEATURE_NUM,), dtype=np.float32)
        self.num_classes = constants.CLASS_NUM
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        self.random = random
        
    
    def step(self, action):
        self.episode_length += 1
        reward = 0
        if self.episode_length == self.max_steps: 
            reward -=1
            self.total_reward -=1
            terminated = True
            done = True
            y_actual = self.y
            y_pred = np.nan
            is_success = False
        elif action < self.num_classes: 
            if action == self.y:
                reward +=1
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
        elif self.actions[action] in self.trajectory: 
            terminated = False
            reward -= 1
            self.total_reward -= 1
            done = False
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        else: 
            terminated = False
            reward += 1
            self.total_reward += 1
            done = False
            self.state = self.get_next_state(action-self.num_classes)
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        self.trajectory.append(self.actions[action])
        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward': self.total_reward, 'y_pred': y_pred, 
                'y_actual': y_actual, 'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}
        return self.state, reward, done, info
            
    
    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')
        
            
    
    def reset(self):
        if self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx == len(self.X):
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.state = np.zeros((constants.FEATURE_NUM,), dtype=np.float32)
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


class SyntheticPPOEnv(Env):
    def __init__(self, X, Y, random=True):
        super(SyntheticPPOEnv, self).__init__()
        self.action_space = Discrete(constants.ACTION_NUM)
        self.observation_space = Box(0, 1.5, (constants.FEATURE_NUM,))
        #self.actions = constants.ACTION_SPACE
        self.actions = constants.ACTION_SPACE
        self.max_steps = constants.MAX_STEPS
        self.X = X
        self.Y = Y
        self.sample_num = len(X)
        self.idx = -1
        self.x = np.zeros((constants.FEATURE_NUM,), dtype=np.float32)
        self.y = np.nan
        self.state = np.zeros((constants.FEATURE_NUM,), dtype=np.float32)
        self.num_classes = constants.CLASS_NUM
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        self.random = random
        
    
    def step(self, action):
        self.episode_length += 1
        reward = 0
        if self.episode_length == self.max_steps: 
            reward -=1
            self.total_reward -=1
            terminated = True
            done = True
            y_actual = self.y
            y_pred = np.nan
            is_success = False
        elif action < self.num_classes: 
            if action == self.y:
                reward +=1
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
        elif self.actions[action] in self.trajectory: 
            terminated = False
            reward -= 1
            self.total_reward -= 1
            done = False
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        else: 
            terminated = False
            reward += 1
            self.total_reward += 1
            done = False
            self.state = self.get_next_state(action-self.num_classes)
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
        self.trajectory.append(self.actions[action])
        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward': self.total_reward, 'y_pred': y_pred, 
                'y_actual': y_actual, 'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}
        return self.state, reward, done, info
            
    
    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')
        
            
    
    def reset(self):
        if self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx == len(self.X):
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.state = np.zeros((constants.FEATURE_NUM,), dtype=np.float32)
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

# class MimicEnv(Env):
#     def __init__(self, X, Y, random=True):
#         super(MimicEnv, self).__init__()
#         self.action_space = Discrete(constants.MIMIC_ACTION_NUM)
#         self.observation_space = Box(0, 1.5, (constants.MIMIC_FEATURE_NUM,))
#         #self.actions = constants.ACTION_SPACE
#         self.actions = constants.MIMIC_ACTION_SPACE
#         self.max_steps = constants.MIMIC_MAX_STEPS
#         self.X = X
#         self.Y = Y
#         self.sample_num = len(X)
#         self.idx = -1
#         self.x = np.zeros((constants.MIMIC_FEATURE_NUM,), dtype=np.float32)
#         self.y = np.nan
#         self.state = np.zeros((constants.MIMIC_FEATURE_NUM,), dtype=np.float32)
#         self.num_classes = constants.MIMIC_CLASS_NUM
#         self.episode_length = 0
#         self.trajectory = []
#         self.total_reward = 0
#         self.random = random
        
    
#     def step(self, action):
#         self.episode_length += 1
#         reward = 0
#         if self.episode_length == self.max_steps: 
#             reward -=1
#             self.total_reward -=1
#             terminated = True
#             done = True
#             y_actual = self.y
#             y_pred = np.nan
#             is_success = False
#         elif action < self.num_classes: 
#             if action == self.y:
#                 reward +=1
#                 self.total_reward += 1
#                 is_success = True
#             else:
#                 reward -= 1
#                 self.total_reward -= 1
#                 is_success = False
#             terminated = False
#             done = True
#             y_actual = self.y
#             y_pred = action
#         elif self.actions[action] in self.trajectory: 
#             terminated = False
#             reward -= 1
#             self.total_reward -= 1
#             done = False
#             y_actual = np.nan
#             y_pred = np.nan
#             is_success = None
#         else: 
#             terminated = False
#             reward += 1
#             self.total_reward += 1
#             done = False
#             self.state = self.get_next_state(action-self.num_classes)
#             y_actual = np.nan
#             y_pred = np.nan
#             is_success = None
#         self.trajectory.append(self.actions[action])
#         info = {'index': self.idx, 'episode_length':self.episode_length, 'reward': self.total_reward, 'y_pred': y_pred, 
#                 'y_actual': y_actual, 'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}
#         return self.state, reward, done, info
            
    
#     def render(self):
#         print(f'STEP {self.episode_length} for index {self.idx}')
#         print(f'Current state: {self.state}')
#         print(f'Total reward: {self.total_reward}')
#         print(f'Trajectory: {self.trajectory}')
        
            
    
#     def reset(self):
#         if self.random:
#             self.idx = random.randint(0, self.sample_num-1)
#         else:
#             self.idx += 1
#             if self.idx == len(self.X):
#                 raise StopIteration()
#         self.x, self.y = self.X[self.idx], self.Y[self.idx]
#         self.state = np.zeros((constants.MIMIC_FEATURE_NUM,), dtype=np.float32)
#         self.trajectory = []
#         self.episode_length = 0
#         self.total_reward = 0
#         return self.state
        
    
#     def get_next_state(self, feature_idx):
#         self.x = self.x.reshape(-1, constants.MIMIC_FEATURE_NUM)
#         x_value = self.x[0, feature_idx]
#         next_state = copy.deepcopy(self.state)
#         next_state[feature_idx] = x_value
#         return next_state




if __name__ =='__main__':
    print('SIMPLE ENV')
    df = pd.read_csv('data/dataset_10000.csv')
    class_dict = {'A':0, 'B':1, 'C':2}
    df['label'] = df['label'].replace(class_dict)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    X, y = np.array(X), np.array(y)
    env = SyntheticSimpleEnv(X, y)
    print(env.actions)

    print('COMPLEX ENV')
    df = pd.read_csv('data/anemia_synth_dataset.csv')
    df = df.fillna(0)
    classes = list(df.label.unique())
    nums = [i for i in range(len(classes))]
    class_dict = dict(zip(classes, nums))
    df['label'] = df['label'].replace(class_dict)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    X, y = np.array(X), np.array(y)
    env = SyntheticComplexEnv(X, y)
    print(env.actions)
