import os
import random
import numpy as np
import constants
from gym import Envfrom gym.spaces import Discrete, Box



random.SEED(constants.SEED)
np.random.SEED(constants.SEED)
os.environ['PYTHONHASHSEED']=str(SEED)


class LupusEnv(Env):
