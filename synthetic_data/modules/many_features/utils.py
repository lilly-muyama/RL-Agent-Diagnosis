import pandas as pd 
import numpy as np 
import seaborn as sns
import random
import os
import ast
import torch
from modules.many_features import constants
from modules.many_features.env import SyntheticEnv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from datetime import datetime


random.seed(constants.SEED)
np.random.seed(constants.SEED)
os.environ['PYTHONHASHSEED']=str(constants.SEED)
torch.manual_seed(constants.SEED)
torch.use_deterministic_algorithms(True)


def load_dqn3(filename, env=None):
    from stable_baselines3 import DQN
    print('Using stable baselines 3')
    model = DQN.load(filename, env=env)
    return model

def load_dqn(filename, env=None):
    from stable_baselines import DQN
    print('Using just stable baselines (not 3)')
    model = DQN.load(filename, env=env)
    return model

def create_env(X, y, random=True):
    env = SyntheticEnv(X, y, random)
    return env

def stable_dqn3(X_train, y_train, timesteps, save=False, filename=None, checkpoint_folder=None, checkpoint_prefix = 'dqn3_basic'):
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback
    print('using stable baselines 3')
    torch.manual_seed(constants.SEED)
    torch.use_deterministic_algorithms(True)

    env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', env, verbose=1, seed=constants.SEED)
    checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=checkpoint_folder, name_prefix=checkpoint_prefix)
    model.learn(total_timesteps=timesteps, log_interval=100000, callback=checkpoint_callback)
    if save:
        model.save(filename)
    env.close()
    return model


def stable_dqn(X_train, y_train, timesteps, save=False, filename=None, checkpoint_folder=None, checkpoint_prefix = 'dqn_basic'):
    from stable_baselines import DQN
    from stable_baselines import bench, logger
    from stable_baselines.common.callbacks import CheckpointCallback
    import tensorflow
    tensorflow.set_random_seed(constants.SEED)

    print('using just stable baselines (not 3)')
    training_env = create_env(X_train, y_train)
    # training_env = bench.Monitor(training_env, logger.get_dir())
    # model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, n_cpu_tf_sess=1)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1)
    # checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=checkpoint_folder, name_prefix=checkpoint_prefix)
    model.learn(total_timesteps=timesteps, log_interval=10000)#, callback=checkpoint_callback)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model
    
def stable_dueling_dqn(X_train, y_train, timesteps, save=False, filename=None, checkpoint_folder=None, checkpoint_prefix = 'dqn_basic'):
    from stable_baselines import DQN
    from stable_baselines import bench, logger
    from stable_baselines.common.callbacks import CheckpointCallback
    import tensorflow
    tensorflow.set_random_seed(constants.SEED)

    print('using just stable baselines (not 3)')
    training_env = create_env(X_train, y_train)
    # training_env = bench.Monitor(training_env, logger.get_dir())
    # model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, n_cpu_tf_sess=1)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, double_q=False, prioritized_replay=True)
    # checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=checkpoint_folder, name_prefix=checkpoint_prefix)
    model.learn(total_timesteps=timesteps, log_interval=10000)#, callback=checkpoint_callback)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model

def stable_double_dqn(X_train, y_train, timesteps, save=False, filename=None, checkpoint_folder=None, checkpoint_prefix = 'ddqn_basic'):
    from stable_baselines import DQN
    from stable_baselines import bench, logger
    from stable_baselines.common.callbacks import CheckpointCallback
    import tensorflow
    tensorflow.set_random_seed(constants.SEED)

    print('using just stable baselines (not 3)')
    training_env = create_env(X_train, y_train)
    # training_env = bench.Monitor(training_env, logger.get_dir())
    # model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, n_cpu_tf_sess=1)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, policy_kwargs=dict(dueling=False),
                prioritized_replay=True)
    # checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=checkpoint_folder, name_prefix=checkpoint_prefix)
    model.learn(total_timesteps=timesteps, log_interval=10000)#, callback=checkpoint_callback)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model

def stable_vanilla_dqn(X_train, y_train, timesteps, save=False, filename=None, checkpoint_folder=None, checkpoint_prefix = 'vanilla_dqn'):
    from stable_baselines import DQN
    from stable_baselines import bench, logger
    from stable_baselines.common.callbacks import CheckpointCallback
    import tensorflow
    tensorflow.set_random_seed(constants.SEED)

    print('using just stable baselines (not 3)')
    training_env = create_env(X_train, y_train)
    # training_env = bench.Monitor(training_env, logger.get_dir())
    # model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, n_cpu_tf_sess=1)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, policy_kwargs=dict(dueling=False),
                double_q=False, prioritized_replay=True)
    # checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpoint_folder, name_prefix=checkpoint_prefix)
    model.learn(total_timesteps=timesteps, log_interval=50000)#, callback=checkpoint_callback)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model

def stable_prioritized_dqn(X_train, y_train, timesteps, save=False, filename=None, checkpoint_folder=None, checkpoint_prefix = 'vanilla_dqn'):
    from stable_baselines import DQN
    from stable_baselines import bench, logger
    from stable_baselines.common.callbacks import CheckpointCallback
    import tensorflow
    tensorflow.set_random_seed(constants.SEED)

    print('using just stable baselines (not 3)')
    training_env = create_env(X_train, y_train)
    # training_env = bench.Monitor(training_env, logger.get_dir())
    # model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, n_cpu_tf_sess=1)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, prioritized_replay=True)
    # checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=checkpoint_folder, name_prefix=checkpoint_prefix)
    model.learn(total_timesteps=timesteps, log_interval=10000)#, callback=checkpoint_callback)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model
    

def evaluate_dqn(dqn_model, X_test, y_test):
    test_df = pd.DataFrame()
    env = create_env(X_test, y_test, random=False)
    count=0

    try:
        while True:
            count+=1
            # if count%(len(X_test)/5)==0:
            #     print(f'Count: {count}')
            obs, done = env.reset(), False
            while not done:
                action, _states = dqn_model.predict(obs, deterministic=True)
                obs, rew, done, info = env.step(action)
                if done == True:
                    #print(f'info: {info}')
                    test_df = test_df.append(info, ignore_index=True)
    except StopIteration:
        print('Testing done.....')
    return test_df

def diagnose_sample(dqn_model, X_test, y_test, idx):

    env = create_env(X_test, y_test, random=False)
    try:
        obs, done = env.reset(idx=idx), False
        start = datetime.now()
        while not done:
            action, _states = dqn_model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(action)
            if done==True:
                end = datetime.now()
                duration = end-start
                return duration, info['trajectory']
    except Exception as e:
        print(e)

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

# def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
#     '''Calculate roc_auc score'''
#     fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
#     target= list(class_dict.keys())
#     lb = LabelBinarizer()
#     lb.fit(y_test)
#     y_test = lb.transform(y_test)
#     y_pred = lb.transform(y_pred)

#     for (idx, c_label) in enumerate(target):
#         fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
#         c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
#     c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
#     plt.close()
#     return roc_auc_score(y_test, y_pred, average=average)

def test(ytest, ypred):
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, average ='macro', labels=np.unique(ytest))
    try:
        roc_auc = multiclass(ytest, ypred)
    except:
        roc_auc = None
    return acc*100, f1*100, roc_auc*100

def test_dt(model, Xtest, ytest):
    ypred = model.predict(Xtest)
    acc = accuracy_score(ytest, ypred)
    f1_macro = f1_score(ytest, ypred, average ='macro', labels=np.unique(ytest))
    cr = classification_report(ytest, ypred)
    cm = confusion_matrix(ytest, ypred)
    roc_auc = multiclass(ytest, ypred)
    return acc, f1_macro, cr, cm, roc_auc, ypred

def get_avg_length_reward(df):
    length = np.mean(df.episode_length)
    reward = np.mean(df.reward)
    return length, reward

def success_rate(test_df):
    success_df = test_df[test_df['y_pred']==test_df['y_actual']]
    success_rate = len(success_df)/len(test_df)*100
    return success_rate, success_df

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

def balance_dataset(df, num=None):
    balanced_df = pd.DataFrame()
    if num is None:
        num = df.label.value_counts().iloc[-1]
    for label in df.label.unique():
        lb_df = df[df.label==label]
        if df.label.value_counts().loc[label] <= num:
            balanced_df = balanced_df.append(lb_df)
        else:
            balanced_df = balanced_df.append(lb_df[:num])
            
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    return balanced_df

def generate_nans(df, column_list, frac):
    #simulating missing values in the data, frac can be a float or a list of floats same length as column_list
    for i, col in enumerate(column_list):
        if isinstance(frac, float):
            vals_to_nan = df[col].dropna().sample(frac=frac, random_state=42).index
        elif instance(frac, list) & (len(column_list)==len(frac)):
            vals_to_nan = df[col].dropna().sample(frac=frac[i]).index
        elif len(column_list) != len(frac):
            print('The column and frac lists should be of the same length')
            return
        else:
            print('I have no idea what is happening :)')
            return
        df.loc[vals_to_nan, col] = np.nan
    return df

def get_dt_performance(df, labels='string'): #labels can also be numeric
    #performance of a decision tree on a dataset
    df = df.fillna(-1)
    if labels == 'string':
        class_dict = constants.CLASS_DICT
        df['label'] = df['label'].replace(class_dict)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=constants.SEED)
    dt = DecisionTreeClassifier(random_state=constants.SEED).fit(X_train, y_train)
    acc, f1, cr, cm, roc_auc, y_pred  = test_dt(dt, X_test, y_test) 
    start= datetime.now()
    dt.predict(np.array(X_test)[0].reshape(1, -1))
    end = datetime.now()
    duration = end-start
    return acc, f1, roc_auc, duration

##############################################PATHWAY FUNCTIONS#########################################################################
get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))

anemias = list(constants.CLASS_DICT.keys())

def generate_filename(i):
    anemia = anemias[i]
    filename = anemia.lower().replace(' ', '_').replace('/','_')
    return filename

def generate_title(i, patient_num):
    anemia = anemias[i]
    title = f'Diagnosis Pathway for {anemia} - ({patient_num} patients)'
    return title

def generate_tuple_dict(df):
    frequency_dict = {}
    for traj in df.trajectory:
        if traj in frequency_dict.keys():
            frequency_dict[traj] += 1
        else:
            frequency_dict[traj] = 1
    #print(f'frequency_dict: {frequency_dict}')
    overall_tup_dict = {}
    for key, value in frequency_dict.items():
        new_key = ast.literal_eval(key)
        for tup in zip(new_key, new_key[1:]):
            #print(f'tup: {tup}')
            if tup in overall_tup_dict.keys():
                overall_tup_dict[tup] += value
            else:
                overall_tup_dict[tup] = value
    #print(f'overall_tup_dict: {overall_tup_dict}')
    return overall_tup_dict

def create_sankey_df(df):
    overall_tup_dict = generate_tuple_dict(df)
    sankey_df = pd.DataFrame()
    sankey_df['Label1'] = [i[0] for i in overall_tup_dict.keys()]
    sankey_df['Label2'] = [i[1] for i in overall_tup_dict.keys()]
    sankey_df['value'] = list(overall_tup_dict.values())
    return sankey_df


def create_source_and_target(sankey_df, dmap):
    sankey_df['source'] = sankey_df['Label1'].map(dmap)
    sankey_df['target'] = sankey_df['Label2'].map(dmap)
    sankey_df.sort_values(by=['source'], inplace=True)
    return sankey_df

def draw_sankey_diagram(pos_df, neg_df, title, save=False, filename=False):
    pos_sankey_df = create_sankey_df(pos_df)
    neg_sankey_df = create_sankey_df(neg_df)
    unique_actions = list(set(list(pos_sankey_df['Label1'].unique()) + list(pos_sankey_df['Label2'].unique()) + list(neg_sankey_df['Label1'].unique()) + list(neg_sankey_df['Label2'].unique())))
    dmap = dict(zip(unique_actions, range(len(unique_actions))))
    
    pos_sankey_df = create_source_and_target(pos_sankey_df, dmap)
    neg_sankey_df = create_source_and_target(neg_sankey_df, dmap)
    #nodes_color = get_colors(len(dmap))
    nodes_color = 'orange'
    
    label = unique_actions
    
    target = list(pos_sankey_df['target']) + list(neg_sankey_df['target'])
    value = list(pos_sankey_df['value']) + list(neg_sankey_df['value'])
    source = list(pos_sankey_df['source']) + list(neg_sankey_df['source'])
    link_color = ['green']*len(pos_sankey_df) + ['red']*len(neg_sankey_df)
#     layout = go.Layout(

# )
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=label, color=nodes_color),
        link= dict(source=source, target=target, value=value, color=link_color)
    )])
    fig.update_layout(title_text=title, 
                      title_x=0.5,  
                      title_font_size=24, 
                      title_font_color='black', 
                      title_font_family='Times New Roman', 
                      font = dict(family='Times New Roman', size=38),
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      plot_bgcolor='rgba(0, 0, 0, 0)'
                      )
    
    if save:
        fig.write_html(f'{filename}.html')
    fig.show()

def draw_sankey_diagram_orig(df, title, save=False, filename=False):
    overall_tuple_dict = generate_tuple_dict(df)
    sankey_df = pd.DataFrame()
    sankey_df['Label1'] = [i[0] for i in overall_tuple_dict.keys()]
    sankey_df['Label2'] = [i[1] for i in overall_tuple_dict.keys()]
    sankey_df['value'] = list(overall_tuple_dict.values())
    unique_actions = list(set(list(sankey_df['Label1'].unique())  + list(sankey_df['Label2'].unique())))
    dmap = dict(zip(unique_actions, range(len(unique_actions))))
    sankey_df['source'] = sankey_df['Label1'].map(dmap)
    sankey_df['target'] = sankey_df['Label2'].map(dmap)
    sankey_df.sort_values(by=['source'], inplace=True)
    nodes_color = get_colors(len(dmap))
    label = unique_actions
    target = list(sankey_df['target'])
    value = list(sankey_df['value'])
    source = list(sankey_df['source'])
    link_color = get_colors(len(value))
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=label, color=nodes_color),
        link = dict(source=source, target=target, value=value, color=link_color)
    )])
    fig.update_layout(title_text=title, title_x=0.5,  title_font_size=24, title_font_color='black', 
                      title_font_family='Times New Roman')
    if save:
        fig.write_html(f'{filename}.html')
    fig.show()


############################CONFUSION MATRIX AND CLASSIFICATION REPORT FUNCTIONS##########################################################
def plot_confusion_matrix(y_actual, y_pred, save=False, filename=False):
    cm = confusion_matrix(y_actual, y_pred)
    cm_df = pd.DataFrame(cm, index = constants.CLASS_DICT.keys(), columns = constants.CLASS_DICT.keys())
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_df, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Anemia')
    plt.xlabel('Predicted Anemia')
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()
    plt.close()


def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def show_values(pc, fmt="%.2f", **kw):    
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))

def plot_classification_report(y_actual, y_pred, save=False, filename=False, cmap='RdBu'):
    cr = classification_report(y_actual, y_pred)
    lines = cr.split('\n')
    class_names = list(constants.CLASS_DICT.keys())
    plotMat = []
    support = []
    #class_names = []
    #count = 0
    for line in lines[2 : (len(lines) - 5)]:
        t = line.strip().split()
        if len(t) < 2: continue
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        plotMat.append(v)

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    ytick_labels = [f'{class_names[i]}({sup})' for i, sup in enumerate(support) ]
    
    #print(len(support))
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), 'classification report', xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)
    #plt.tight_layout()
    if save:
        plt.savefig(filename, bbox_inches = 'tight')
    plt.show()
    plt.close()

