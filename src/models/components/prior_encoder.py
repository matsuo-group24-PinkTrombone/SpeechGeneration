import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions.normal import Normal

from ..abc._types import _t_or_any, _tensor_or_any
from ..abc.prior import Prior as AbstractPrior
from .linear_layers import LinearLayers


class Prior(AbstractPrior):
    def __init__(self, hidden_dim: int, state_dim: int, min_stddev: float = 0.1) -> None:
        """
        Args:
            hidden_dim (int): The length of Hidden dimension
            state_dim (int): The length of State dimension
            min_stddev (float, optional): inimum value of standard deviation. Defaults to 0.1.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.min_stddev = min_stddev

        assert hidden_dim > 0, "hidden_dim must be greater than 0"
        assert state_dim > 0, "state_dim must be greater than 0"

        self.fc_to_mean = LinearLayers(
            input_size=hidden_dim, hidden_size=hidden_dim * 2, layers=3, output_size=state_dim
        )
        self.fc_to_stddev = LinearLayers(
            input_size=hidden_dim, hidden_size=hidden_dim * 2, layers=3, output_size=state_dim
        )

    def forward(self, hidden: _tensor_or_any) -> _t_or_any[Distribution]:
        """
        Args:
            hidden (_tensor_or_any): Hidden vectors computed deterministically.

        Returns:
            _t_or_any[Distribution]: The Normal distribution
        """
        mean = self.fc_to_mean(hidden)
        stddev = (
            F.softplus(self.fc_to_stddev(hidden)) + self.min_stddev
        )  # Follows the code of exercise7
        return Normal(mean, stddev)

    @property
    def state_shape(self) -> tuple[int]:
        """The getter of the shape of state.

        Returns:
            tuple[int]: The size of state dimension.
        """
        return (self.state_dim,)
