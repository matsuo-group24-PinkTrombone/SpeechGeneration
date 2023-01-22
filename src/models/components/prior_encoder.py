import torch
import torch.nn as nn
from torch.distributions import Distribution
from torch.distributions.normal import Normal

from ..abc import prior
from ..abc._types import _t_or_any, _tensor_or_any


class Prior(prior.Prior):
    def __init__(self, hidden_dim: int, state_dim: int) -> None:
        assert hidden_dim > 0, "hidden_dim must be greater than 0"
        assert state_dim > 0, "state_dim must be greater than 0"
        self.fc_to_mean = nn.Linear(hidden_dim, state_dim)
        self.fc_to_var = nn.Linear(hidden_dim, state_dim)

    def forward(self, hidden: _tensor_or_any) -> _t_or_any[Distribution]:
        mean = self.fc_to_mean(hidden)
        var = self.fc_to_var(hidden)
        return Normal(mean, var)
