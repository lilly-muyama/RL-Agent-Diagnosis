import random
import constants
import pandas as pd
from envs import SyntheticComplexHbEnv

#random.seed(constants.SEED)

class RandomAgent():
    def __init__(self, X, y):
        self.action_space = constants.ACTION_SPACE
        self.n_actions = constants.ACTION_NUM
        self.env = SyntheticComplexHbEnv(X, y, random=False)


    def get_action(self):
        action = random.choice(list(range(self.n_actions)))
        return action

    def test(self):
        test_df = pd.DataFrame()
        try:
            while True:
                obs, done = self.env.reset(), False
                while not done:
                    action = self.get_action()
                    obs, rew, done, info = self.env.step(action)
                    if done == True:
                        test_df = test_df.append(info, ignore_index=True)
        except StopIteration:
            print('Testing done.....')
        return test_df