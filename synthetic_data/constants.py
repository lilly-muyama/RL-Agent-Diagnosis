# anemia_synth_dataset_hb
# ACTION_SPACE = ['Aplastic anemia', 'No anemia', 'Hemolytic anemia', 'Anemia of chronic disease', 'Iron deficiency anemia', 
# 'Vitamin B12/Folate deficiency anemia', 'hemoglobin', 'ferritin', 'ret_count', 'segmented_neutrophils', 'tibc', 'mcv']

# anemia_synth_dataset_hb_some_nans  The real deal
# ACTION_SPACE = ['No anemia', 'Hemolytic anemia', 'Aplastic anemia', 'Iron deficiency anemia', 'Vitamin B12/Folate deficiency anemia', 
# 'Anemia of chronic disease', 'hemoglobin', 'ferritin', 'ret_count', 'segmented_neutrophils', 'tibc', 'mcv']
SEED = 42
# ACTION_NUM = len(ACTION_SPACE)
FEATURE_NUM = 6
# CLASS_NUM = 6
MAX_STEPS = 8

#noisy_dataset
# NOISY_ACTION_SPACE = ['Aplastic anemia', 'Hemolytic anemia', 'No anemia', 'Anemia of chronic disease', 'Vitamin B12/Folate deficiency anemia',
# 'Iron deficiency anemia', 'hemoglobin', 'ferritin', 'ret_count', 'segmented_neutrophils', 'tibc', 'mcv']

#noisy_data_uniform_all_30_08_22.csv
# NOISY_ACTION_SPACE = ['Hemolytic anemia', 'Anemia of chronic disease', 'No anemia', 'Aplastic anemia', 
# 'Vitamin B12/Folate deficiency anemia','Iron deficiency anemia', 'hemoglobin', 'ferritin', 'ret_count', 'segmented_neutrophils', 'tibc',
# 'mcv']

# MIMIC_ACTION_SPACE = ['Iron deficiency anemia', 'Vitamin B12/Folate deficiency anemia', 'Hemolytic anemia', 'Aplastic anemia', 
# 'Anemia of chronic disease', 'No anemia', 'max_ret_count', 'mean_ret_count', 'min_ret_count', 'max_ferritin', 'mean_ferritin',
# 'min_ferritin', 'max_hemoglobin', 'mean_hemoglobin', 'min_hemoglobin', 'max_iron', 'mean_iron', 'min_iron', 'max_mcv', 'mean_mcv', 
# 'min_mcv', 'max_rbc', 'mean_rbc', 'min_rbc', 'max_segmented_neutrophils', 'mean_segmented_neutrophils', 'min_segmented_neutrophils',
# 'max_tibc', 'mean_tibc', 'min_tibc', 'age', 'gender']
# MIMIC_ACTION_NUM = len(MIMIC_ACTION_SPACE)
# MIMIC_FEATURE_NUM = 26
# MIMIC_CLASS_NUM = 6
# MIMIC_MAX_STEPS = 14

#anemia_synth_dataset_with_unspecified
ACTION_SPACE = ['No anemia', 'Hemolytic anemia', 'Aplastic anemia', 'Iron deficiency anemia', 'Vitamin B12/Folate deficiency anemia',
'Anemia of chronic disease', 'Unspecified anemia', 'hemoglobin', 'ferritin', 'ret_count', 'segmented_neutrophils', 'tibc', 'mcv']
ACTION_NUM = len(ACTION_SPACE)
CLASS_NUM = 7

