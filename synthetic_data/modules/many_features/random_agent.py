import random
import constants
import pandas as pd
#from env import SyntheticEnv
from modules.many_features.env import SyntheticEnv

#random.seed(constants.SEED)

class RandomAgent():
    def __init__(self, X, y):
        self.action_space = constants.ACTION_SPACE
        self.n_actions = constants.ACTION_NUM
        self.env = SyntheticEnv(X, y, random=False)


    def get_action(self):
        action = random.choice(list(range(self.n_actions)))
        return action

    def test(self):
        test_df = pd.DataFrame()
        try:
            while True:
                #print(f'Resetting environment ....')
                obs, done = self.env.reset(), False
                #print(f'index: {self.env.idx}')
                #print(f'state: {obs}')
                while not done:
                    action = self.get_action()
                    #print(f'action: {action}, action_name: {self.env.actions[action]}')
                    obs, rew, done, info = self.env.step(action)
                    #print(f'new state: {obs}')
                    #print(f'reward: {rew}')
                    #print(f'done: {done}')
                    #print(f'info: {info}')
                    if done == True:
                        #print('Appending to test df ....')
                        test_df = test_df.append(info, ignore_index=True)
        except StopIteration:
            print('Testing done.....')
        return test_df
    
    def test_sample(self, idx):
        try:
            obs, done = self.env.reset(idx=idx), False
            while not done:
                action = self.get_action()
                obs, rew, done, info = self.env.step(action)
                if done==True:
                    return info['trajectory']
        except Exception as e:
            print(e)
