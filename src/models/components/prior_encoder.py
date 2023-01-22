import torch
import torch.nn as nn
from torch.distributions import Distribution

from ..abc import prior
from ..abc._types import _t_or_any, _tensor_or_any

class Prior(prior.Prior):
    def __init__(self, hidden_dim: int, state_dim:int)-> None:
        pass

    def forward(self, hidden: _tensor_or_any) -> _t_or_any[Distribution]:
        pass
