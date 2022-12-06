import pandas as pd
import numpy as np
from modules import constants
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

domains_feat_dict = constants.DOMAINS_FEAT_DICT
domains_max_scores_dict = constants.DOMAINS_MAX_SCORES_DICT
criteria_weights = constants.CRITERIA_WEIGHTS

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
    df = pd.DataFrame(columns = constants.ACTION_SPACE[constants.CLASS_NUM:])
    df = df.append(state)
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

def test(ytest, ypred): # changed to show precision and recall
    accuracy = accuracy_score(ytest, ypred)
    precision = precision_score(ytest, ypred)
    recall = recall_score(ytest, ypred)
    f1 = f1_score(ytest, ypred)
    return accuracy*100, precision*100, recall*100, f1*100

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

