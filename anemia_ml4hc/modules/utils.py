import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from modules.env import AnemiaEnv 
from modules import constants


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

def get_metrics(test_df):
    ytest, ypred = test_df['y_actual'], test_df['y_pred']
    acc = accuracy_score(ytest, ypred)*100
    f1 = f1_score(ytest, ypred, average ='macro', labels=np.unique(ytest))*100
    try:
        roc_auc = multiclass(ytest, ypred)*100
    except:
        roc_auc = None
    return acc, f1, roc_auc

def create_env(X, y, random=True):
    env = AnemiaEnv(X, y, random)
    return env

def stable_dqn(X_train, y_train, timesteps, log_interval=100000, save=False, filename=None):
    from stable_baselines import DQN
    from stable_baselines import bench, logger
    import tensorflow
    tensorflow.set_random_seed(constants.SEED)

    print('using just stable baselines (not 3)')
    training_env = create_env(X_train, y_train)
    training_env = bench.Monitor(training_env, logger.get_dir())
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, n_cpu_tf_sess=1)
    # model.learn(total_timesteps=timesteps, log_interval=log_interval)
    # if save:
    #     model.save(f'{filename}.pkl')
    training_env.close()
    return model