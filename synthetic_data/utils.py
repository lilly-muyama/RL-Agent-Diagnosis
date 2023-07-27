import pandas as pd
import numpy as np
import random
import tensorflow
import os
import constants
import d3rlpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, roc_curve
from envs import SyntheticSimpleEnv, SyntheticComplexEnv
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DQN
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tensorflow.set_random_seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)

def split_dataset(filename, class_dict):
    '''
    Splits dataset into train/test splits
    '''
    df = pd.read_csv(filename)
    if df.isnull().values.any():
        df = df.fillna(0)
    df['label'] = df['label'].replace(class_dict)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    return X_train, X_test, y_train, y_test

def create_d3rlpy_dataset(policy, env, filename): #filename e.g. data/random_policy_dataset.h5
    '''
    Creates and saves an offline dataset using a given policy in a given environment
    '''
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)
    if policy=='random':
        random_policy = d3rlpy.algos.DiscreteRandomPolicy()
        random_policy.collect(env, buffer, n_steps=120000)
    elif policy == 'dqn':
        dqn = d3rlpy.algos.DQN()
        explorer = d3rlpy.online.explorers.ConstantEpsilonGreedy(0.3)
        dqn.fit_online(env, buffer, n_steps=120000)
    else:
        return
    dataset = buffer.to_mdp_dataset()
    dataset.dump(filename)
    return dataset

def train_d3rlpy_dqn(dataset, env, model_name, n_epochs):
    '''
    Trains a dqn using an offline dataset
    '''
    dataset = MDPDataset.load(dataset)
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.1, random_state=SEED)
    dqn = DQN(use_gpu = False)
    dqn.build_with_dataset(dataset)
    td_error = td_error_scorer(dqn, test_episodes)
    evaluate_scorer = evaluate_on_environment(env)
    rewards = evaluate_scorer(dqn)
    dqn.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=10,
            scorers={
                'td_error': td_error_scorer,
                'value_scale': average_value_estimation_scorer,
                'environment': evaluate_scorer
            })
    dqn.save_model(f'{model_name}.pt')
    return dqn

def test_d3rlpy_dqn(dqn_model, env):
    test_df = pd.DataFrame()
    try:
        while True:
            obs, done = env.reset(), False
            if env.idx%1000 == 0:
                print(env.idx)
            while not done:
                #print(f'obs: {obs}')
                action = dqn_model.predict([obs])[0]
                #print(f'action:{action}')
                obs, rew, done, info = env.step(action)
                #print(f'new obs: {obs}')
                #print(f'reward: {rew}')
                #print(f'done: {done}')
                #print(f'info: {info}')
                if done == True:
                    test_df = test_df.append(info, ignore_index=True)
    except StopIteration:
        print('Testing done.....')
    return test_df

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

def test(ytest, ypred):
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, average ='macro', labels=np.unique(ytest))
    try:
        roc_auc = multiclass(ytest, ypred)
    except:
        roc_auc = None
    return acc, f1, roc_auc

def get_success_rate(test_df):
    y_pred_df = test_df[test_df['y_pred'].notna()]
    success_df = y_pred_df[y_pred_df['y_pred']== y_pred_df['y_actual']]
    success_rate = len(success_df)/len(test_df)*100
    return y_pred_df, success_df, success_rate


def get_avg_length_reward(df):
    length = np.mean(df.episode_length)
    reward = np.mean(df.reward)
    return length, reward
