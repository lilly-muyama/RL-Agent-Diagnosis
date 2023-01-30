SEED = 42
# GYM_BOX_LOW = -1
# GYM_BOX_HIGH = np.inf
# CLASS_NUM = 3

# CLASS_DICT = {'No lupus':0, 'Lupus':1, 'Inconclusive diagnosis':2} 
CLASS_DICT = {'No lupus':0, 'lupus':1, 'Inconclusive diagnosis':2}
MAX_STEPS = 20
ACTION_SPACE = ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'fever', 'leukopenia', 'thrombocytopenia', 'auto_immune_hemolysis', 'delirium', 
'psychosis', 'seizure']

ACTION_NUM = len(ACTION_SPACE)
CLASS_NUM = len(CLASS_DICT)
FEATURE_NUM = ACTION_NUM - CLASS_NUM
#MAX_STEPS = 20