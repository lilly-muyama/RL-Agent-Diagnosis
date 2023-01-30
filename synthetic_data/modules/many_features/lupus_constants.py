SEED = 42

CLASS_DICT = {'No lupus':0, 'lupus':1, 'Inconclusive diagnosis':2}

ACTION_SPACE = ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'fever', 'leukopenia', 'thrombocytopenia', 'auto_immune_hemolysis', 'delirium', 
'psychosis', 'seizure', 'non_scarring_alopecia', 'oral_ulcers', 'cutaneous_lupus', 'pleural_effusion', 'pericardial_effusion', 'acute_pericarditis', 
'joint_involvement', 'proteinuria', 'anti_cardioliphin_antibodies', 'anti_β2gp1_antibodies', 'lupus_anti_coagulant', 'low_c3', 'low_c4', 
'anti_dsdna_antibody', 'anti_smith_antibody']

ACTION_NUM = len(ACTION_SPACE)
CLASS_NUM = len(CLASS_DICT)
FEATURE_NUM = ACTION_NUM - CLASS_NUM

MAX_STEPS = FEATURE_NUM+1
