import numpy as np

SEED = 42
FEATURE_NUM = 26
GYM_BOX_LOW = -1
GYM_BOX_HIGH = np.inf
ACTION_SPACE = ['Negative', 'Positive', 'Inconclusive diagnosis', 'ana', 'fever', 'leukopenia', 'thrombocytopenia', 'auto_immune_hemolysis', 'delirium', 
'psychosis', 'seizure', 'non_scarring_alopecia', 'oral_ulcers', 'subacute_cutaneous', 'discoid_lupus', 'acute_cutaneous_lupus', 'pleural_effusion', 
'pericardial_effusion', 'acute_pericarditis', 'joint_involvement', 'proteinuria', 'renal_biopsy_class', 'anti_cardioliphin_antibodies', 
'anti_b2gp4_antibodies', 'lupus_anti_coagulant', 'c3', 'c4', 'anti_dsdna_antibody', 'anti_smith_antibody' ]
MAX_STEPS = 10
CLASS_NUM = 3
ACTION_NUM = FEATURE_NUM + CLASS_NUM
CLASS_DICT = {'Negative':0, 'Positive':1, 'Inconclusive diagnosis':2} 
DOMAINS_FEAT_DICT = {'constitutional': ['fever'], 'hematologic': ['leukopenia', 'thrombocytopenia', 'auto_immune_hemolysis'], 
'neuropsychiatric': ['delirium', 'psychosis', 'seizure'], 
'mucocutaneous': ['non_scarring_alopecia', 'oral_ulcers', 'subacute_cutaneous', 'discoid_lupus', 'acute_cutaneous_lupus'],
'serosal': ['pleural_effusion', 'pericardial_effusion', 'acute_pericarditis'], 'musculoskeletal': ['joint_involvement'], 
'renal': ['proteinuria', 'renal_biopsy_class'], 
'antiphospholipid_antibodies': ['anti_cardioliphin_antibodies', 'anti_b2gp4_antibodies', 'lupus_anti_coagulant'],
'complement_proteins': ['c3', 'c4'], 'sle_specific_antibodies':['anti_dsdna_antibody', 'anti_smith_antibody']}
DOMAINS_MAX_SCORES_DICT = {'constitutional': 2, 'hematologic': 4, 'neuropsychiatric': 5, 'mucocutaneous': 6, 'serosal': 6, 'musculoskeletal': 6, 'renal': 10, 
'antiphospholipid_antibodies': 2, 'complement_proteins': 4, 'sle_specific_antibodies':6}
CRITERIA_WEIGHTS = {'fever':2, 'leukopenia':3, 'thrombocytopenia':4, 'auto_immune_hemolysis':4, 'delirium':2, 'psychosis':3, 'seizure':5, 
'non_scarring_alopecia':2, 'oral_ulcers':2, 'subacute_cutaneous':4, 'discoid_lupus':4, 'acute_cutaneous_lupus':6, 'pleural_effusion': 5, 
'pericardial_effusion':5, 'acute_pericarditis':6, 'joint_involvement':6, 'proteinuria':4, 'renal_biopsy_2_5':8, 'renal_biopsy_3_4': 10, 
'anti_cardioliphin_antibodies':2, 'anti_b2gp4_antibodies':2, 'lupus_anti_coagulant':2, 'low_c3_or_low_c4':3, 'low_c3_and_low_c4':4, 
'anti_dsdna_antibody':6, 'anti_smith_antibody':6,}
