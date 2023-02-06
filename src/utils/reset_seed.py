import random
from typing import Any

import numpy as np
import torch


def reset_seed(seed: Any):
    """Resetting seed.

    Args:
        seed (Any): Seed number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
