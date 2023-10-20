import random
import numpy as np
import torch
from numbers import Number
from typing import Union
from pathlib import Path
import scipy.sparse as sp

data_dir = Path(__file__).parent

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)