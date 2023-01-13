import pandas as pd
import numpy as np
import os
import random
import torch
from modules import constants
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from modules.env import LupusEnv


domains_feat_dict = constants.DOMAINS_FEAT_DICT
domains_max_scores_dict = constants.DOMAINS_MAX_SCORES_DICT
criteria_weights = constants.CRITERIA_WEIGHTS

random.seed(constants.SEED)
np.random.seed(constants.SEED)
os.environ['PYTHONHASHSEED']=str(constants.SEED)

def get_domain_score(row, domain):
    domain_features = domains_feat_dict[domain]
    domain_score = 0
    if domain == 'complement_proteins':
        domain_score = get_c3_c4_score(row.c3, row.c4)
        domain_features = list(set(domain_features) - set(['c3', 'c4']))
    for feat in domain_features:
        if feat == 'renal_biopsy_class': # to delete
            pass
        else:
            if row[feat] >= 0:
                feat_score = get_feat_score(row, feat)
                if feat_score > domain_score:
                    domain_score = feat_score
    if domain_score > domains_max_scores_dict[domain]:
        raise Exception('The score is too large for this domain!')
    return domain_score

def get_c3_c4_score(c3, c4): # 0 - low, 1 is not low, -1 is unknown
    if (c3 == 0) & (c4 == 0):
        return criteria_weights['low_c3_and_low_c4']
    elif (c3 == 0) | (c4 == 0):
        return criteria_weights['low_c3_or_low_c4']
    else:
        return 0

def get_feat_score(row, feat):
    if feat == 'proteinuria':
        feat_score = get_proteinura_score(row[feat])
    elif feat == 'renal_biopsy_class':
        feat_score = get_renal_biopsy_score(row[feat])
    elif row[feat] <= 0:
        feat_score = 0
    else:
        feat_score = criteria_weights[feat]
    return feat_score

def get_proteinura_score(amount):
    if amount > 0.5:
        return 4
    else:
        return 0

def get_renal_biopsy_score(result_class):
    if (result_class == 3) | (result_class == 4):
        return 10
    elif (result_class == 2) | (result_class == 5):
        return 8
    else:
        return 0

def compute_score(state):
    #print(f'state size:{state.shape}')
    #print(f'state type:{type(state)}')
    #print(f'state:{state}')
    state = state.reshape(-1, constants.FEATURE_NUM)
    df = pd.DataFrame(state, columns = constants.ACTION_SPACE[constants.CLASS_NUM:])
    #df = df.append(state)
    row = df.iloc[0]
    if row['ana'] == 0: # negative - 0 positive - 1
        return 0
    total_row_score = 0
    for domain in domains_feat_dict.keys():
        domain_score = get_domain_score(row, domain)
        total_row_score += domain_score
    return total_row_score

def success_rate(test_df):
    success_df = test_df[test_df['y_pred']==test_df['y_actual']]
    success_rate = len(success_df)/len(test_df)*100
    return success_rate, success_df

def get_avg_length_reward(df):
    length = np.mean(df.episode_length)
    reward = np.mean(df.reward)
    return length, reward

def test_binary(ytest, ypred): # changed to show precision and recall
    accuracy = accuracy_score(ytest, ypred)
    precision = precision_score(ytest, ypred)
    recall = recall_score(ytest, ypred)
    f1 = f1_score(ytest, ypred)
    return accuracy*100, precision*100, recall*100, f1*100

def test(ytest, ypred):
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, average ='macro', labels=np.unique(ytest))
    try:
        roc_auc = multiclass(ytest, ypred)
    except:
        roc_auc = None
    return acc*100, f1*100, roc_auc*100

def multiclass(actual_class, pred_class, average = 'macro'):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    avg = sum(roc_auc_dict.values()) / len(roc_auc_dict)
    return avg


def compute_feature_importance(model, x, verbose=False):
    importance = model.feature_importances_
    if verbose:
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(x.columns, importance):
        feats[feature] = importance #add the name/value pair 

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
    importances.sort_values(by='Importance').plot(kind='bar', rot=90)


def split_dataset(df):
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    return X_train, X_test, y_train, y_test

def stable_dqn3(X_train, y_train, timesteps, save=False, filename=None):
    from stable_baselines3 import DQN
    print('using stable baselines 3')
    torch.manual_seed(constants.SEED)
    torch.use_deterministic_algorithms(True)

    env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', env, verbose=1, seed=constants.SEED)
    model.learn(total_timesteps=timesteps, log_interval=100000)
    if save:
        model.save(filename)
    env.close()
    return model

def create_env(X, y, random=True):
    env = LupusEnv(X, y, random)
    return env

def load_dqn3(filename, env=None):
    from stable_baselines3 import DQN
    print('Using stable baselines 3')
    model = DQN.load(filename, env=env)
    return model

def evaluate_dqn(dqn_model, X_test, y_test):
    test_df = pd.DataFrame()
    env = create_env(X_test, y_test, random=False)
    count=0

    try:
        while True:
            count+=1
            if count%(len(X_test)/5)==0:
                print(f'Count: {count}')
            obs, done = env.reset(), False
            while not done:
                #print(f'current state: {obs}')
                action, _states = dqn_model.predict(obs, deterministic=True)
                obs, rew, done, info = env.step(action)
                #print(f'action: {action}')
                #print(f'new state: {obs}')
                #print(f'reward: {rew}')
                #print(f'done: {done}')
                #print(f'info: {info}')

                if done == True:
                    #print(f'info: {info}')
                    test_df = test_df.append(info, ignore_index=True)
                    #print('EPISODE DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    except StopIteration:
        print('Testing done.....')
    return test_df


