import random
import constants
import pandas as pd
from modules.many_features.env import SyntheticEnv
from modules.many_features import constants
#random.seed(constants.SEED)

class DTAgent():
    def __init__(self, X, y):
        self.action_space = constants.ACTION_SPACE
        self.n_actions = constants.ACTION_NUM
        self.env = SyntheticEnv(X, y, random=False)
        self.hb_val = -1


    def get_action(self):
        if len(self.env.trajectory) == 0:
            next_action = 'hemoglobin'
        else:
            last_action = self.env.trajectory[-1]
            next_action = self.predict_next_action(last_action)
        next_action_index = self.env.actions.index(next_action)
        return next_action_index

    def predict_next_action(self, last_action):
        #print(f'The last action: {last_action}')
        # try:
        last_action_idx = self.env.actions.index(last_action)
        #print(f'last action index: {last_action_idx}')
        last_action_val = self.get_feature_value(last_action_idx)
        #print(f'last action value: {last_action_val}')
        # except:
        #     print(f'Invalid last action: {last_action}')
        #     return 
        if last_action == 'hemoglobin':
            self.hb_val = last_action_val
            action = self.predict_act_hb(last_action_val)
            
        elif last_action == 'gender':
            action = self.predict_act_gender(self.hb_val, last_action_val)
        elif last_action == 'mcv':
            action = self.predict_act_mcv(last_action_val)
        elif last_action == 'ret_count':
            action = self.predict_act_ret(last_action_val)
        elif last_action == 'segmented_neutrophils':
            action = self.predict_act_neutrophils(last_action_val)
        elif last_action == 'ferritin':
            action = self.predict_act_ferritin(last_action_val)
        elif last_action == 'tibc':
            action = self.predict_act_tibc(last_action_val)
        else:
            print('Invalid last action')
            raise Exception
        #print(f'Next action: {action}')
        return action

    def get_feature_value(self, idx):
        if idx >= self.env.num_classes:
            feature_idx = idx-self.env.num_classes
            # print(f'Action space length: {len(constants.ACTION_SPACE)}')
            # print(f'Class num: {constants.CLASS_NUM}')
            # print(f'Feature number: {constants.FEATURE_NUM}')
            x = self.env.x.reshape(-1, constants.FEATURE_NUM)
            x_value = self.env.x[0, feature_idx]
            return x_value
        else:
            print('Last action cannot be a diagnosis action')

    def predict_act_hb(self, val):
        if val > 13:
            return 'No anemia'
        elif val< 0:
            print('Hemoglobin can\'t be negative')
            raise Exception
        elif val < 12:
            return 'mcv'
        else:
            return 'gender'

    def predict_act_gender(self, hb_val, gender_val):
        if (hb_val < 0) | (gender_val< 0):
            print('Neither hemoglobin nor gender can be negative')
            raise Exception 
        if (hb_val>= 12) & (gender_val == 0):
            return 'No anemia'
        else:
            return 'mcv'

    def predict_act_mcv(self, val):
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val < 80:
            return 'ferritin'
        elif val <= 100:
            return 'ret_count'
        elif val > 100:
            return 'segmented_neutrophils'
        else:
            return 'Inconclusive diagnosis'

    def predict_act_ret(self, val):
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val <=2:
            return 'Aplastic anemia'
        elif val>2:
            return 'Hemolytic anemia'
        else:
            return 'Inconclusive diagnosis'

    def predict_act_neutrophils(self, val):
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val > 0:
            return 'Vitamin B12/Folate deficiency anemia'
        elif val==0:
            return 'Unspecified anemia'
        else:
            return 'Inconclusive diagnosis'

    def predict_act_ferritin(self, val):
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val < 30:
            return 'Iron deficiency anemia'
        elif val > 100:
            return 'Anemia of chronic disease'
        else:
            return 'tibc'

    def predict_act_tibc(self, val):
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val < 450:
            return 'Anemia of chronic disease'
        elif val >=450:
            return 'Iron deficiency anemia'
        else:
            return 'Inconclusive diagnosis'

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