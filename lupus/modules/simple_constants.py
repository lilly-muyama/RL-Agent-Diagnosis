import numpy as np # to delete
# GYM_BOX_LOW = -1 #to delete
# GYM_BOX_HIGH = np.inf # to delete
# MAX_STEPS=30 #to delete


SEED = 42

#2 FEATURES ANEM DATASET
# ACTION_SPACE = ['No anemia', 'Anemia', 'Inconclusive diagnosis', 'hemoglobin', 'gender']

#2 FEATURES DATASET
# ACTION_SPACE = ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'anti_dsdna_antibody']


#3 FEATURES DATASET
# ACTION_SPACE = ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'anti_dsdna_antibody', 'joint_involvement']

#5 FEATURES DATASET
# ACTION_SPACE = ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'anti_dsdna_antibody', 'joint_involvement', 'proteinuria', 'pericardial_effusion']

#8 FEATURES DATASET
# ACTION_SPACE = ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'anti_dsdna_antibody', 'joint_involvement', 'proteinuria', 'pericardial_effusion',
# 'non_scarring_alopecia', 'leukopenia', 'delirium']

#11 FEATURES DATASET
# ACTION_SPACE = ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'anti_dsdna_antibody', 'joint_involvement', 'proteinuria', 'pericardial_effusion',
# 'non_scarring_alopecia', 'leukopenia', 'delirium', 'low_c3_and_c4', 'fever', 'anti_cardioliphin_antibodies']

#22 FEATURES DATASET
ACTION_SPACE =  ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'non_scarring_alopecia', 'anti_dsdna_antibody', 'joint_involvement', 
'proteinuria', 'pericardial_effusion', 'leukopenia', 'delirium', 'low_c3', 'low_c4', 'fever', 'thrombocytopenia', 'anti_cardioliphin_antibodies', 
'pleural_effusion', 'psychosis', 'seizure', 'lupus_anti_coagulant', 'anti_β2gp1_antibodies', 'anti_smith_antibody',  'oral_ulcers', 
'auto_immune_hemolysis', 'acute_pericarditis'] #cutaneous lupus left out


CLASS_DICT = {'No lupus':0, 'Lupus':1, 'Inconclusive diagnosis':2}
ACTION_NUM = len(ACTION_SPACE)
CLASS_NUM = len(CLASS_DICT)
FEATURE_NUM = ACTION_NUM - CLASS_NUM