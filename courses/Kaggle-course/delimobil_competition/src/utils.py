import os
import random
import numpy as np


def set_random_state(state):
    # python's seeds
    random.seed(state)
    os.environ["PYTHONHASHSEED"] = str(state)
    np.random.seed(state)

    # # torch's seeds
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # # tensorflow's seed
    # tf.random.set_seed(seed)


DEFAULT_RANDOM_STATE = 12345
