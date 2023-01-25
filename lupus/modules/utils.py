import pandas as pd
import numpy as np
import seaborn as sns
import os
import ast
import random
import torch
from modules import constants
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from modules.env import LupusEnv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


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

def get_domain_score(row, domain):
    domain_features = domains_feat_dict[domain]
    domain_score = 0
    if domain == 'complement_proteins':
        domain_score = get_c3_c4_score(row.low_c3, row.low_c4)
        domain_features = list(set(domain_features) - set(['low_c3', 'low_c4']))
    for feat in domain_features:
        if row[feat] >= 0:
            if feat == 'cutaneous_lupus': # to delete
                feat_score = get_cutaneous_lupus_score(row.cutaneous_lupus)
            else:
                feat_score = get_feat_score(row, feat)
            if feat_score > domain_score:
                domain_score = feat_score
    if domain_score > domains_max_scores_dict[domain]:
        raise Exception('The score is too large for this domain!')
    return domain_score

def get_c3_c4_score(c3, c4): # 0 - low, 1 is not low
    if (c3 == 1) & (c4 == 1):
        return criteria_weights['low_c3_and_low_c4']
    elif (c3 == 1) | (c4 == 1):
        return criteria_weights['low_c3_or_low_c4']
    else:
        return 0

def get_feat_score(row, feat):
    if feat == 'cutaneous_lupus':
        feat_score = get_cutaneous_lupus_score(row[feat])
    elif row[feat] <= 0:
        feat_score = 0
    else:
        feat_score = criteria_weights[feat]
    return feat_score

def get_cutaneous_lupus_score(cutaneous_type):
    if cutaneous_type == 0: #negative for any form of cutaneous lupus
        return 0
    elif cutaneous_type == 1: #subacute cutaneous lupus
        return criteria_weights['subacute_cutaneous_lupus']
    elif cutaneous_type == 2: #acute cutaneous lupus
        return criteria_weights['acute_cutaneous_lupus']
    elif cutaneous_type == 3: #discoid lupus
        return criteria_weights['discoid_lupus']
    else:
        print('Unknown cutaneous type')


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

############################CONFUSION MATRIX AND CLASSIFICATION REPORT FUNCTIONS##########################################################
def plot_confusion_matrix(y_actual, y_pred, save=False, filename=False):
    cm = confusion_matrix(y_actual, y_pred)
    if len(y_pred.unique()) == 2:
        cm_df = pd.DataFrame(cm, index = [0, 1], columns = [0, 1])
    elif len(y_pred.unique()) == 3:
        cm_df = pd.DataFrame(cm, index = constants.CLASS_DICT.keys(), columns = constants.CLASS_DICT.keys())
    else:
        print('Unexpected number of predicted classes')
        return
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_df, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
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

##############################################PATHWAY FUNCTIONS#########################################################################
get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))

lupus_classes = list(constants.CLASS_DICT.keys())

def generate_filename(i):
    lupus_class = lupus_classes[i]
    filename = lupus_class.lower().replace(' ', '_').replace('/','_')
    return filename

def generate_title(i, patient_num):
    lupus_class = lupus_classes[i]
    title = f'Diagnosis Pathway for {lupus_class} - ({patient_num} patients)'
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
    # sankey_df.sort_values(by=['value'], inplace=True, ascending=False)  ######### DELETE THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=label, color=nodes_color),
        link= dict(source=source, target=target, value=value, color=link_color)
    )])
    fig.update_layout(title_text=title, title_x=0.5,  title_font_size=24, title_font_color='black', 
                      title_font_family='Times New Roman')
    if save:
        fig.write_html(f'{filename}.html')
    fig.show()


