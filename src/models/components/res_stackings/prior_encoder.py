import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions.normal import Normal

from ...abc.prior import Prior as AbstractPrior
from ..res_layers import ResLayers


class Prior(AbstractPrior):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        stacked_res_hidden_size: int,
        num_layers: int = 1,
        min_stddev: float = 0.1,
        bias: bool = True,
    ) -> None:
        """
        Args:
            hidden_dim (int): The length of Hidden dimension
            state_dim (int): The length of State dimension
            stacked_linear_hidden_size (int): The hidden size of `fc_input_layer`.
            num_layers (int): The number of internal layers (For stacked linear).
            min_stddev (float, optional): inimum value of standard deviation. Defaults to 0.1.
            bias (bool): Bias of `nn.Linear`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.min_stddev = min_stddev

        assert hidden_dim > 0, "hidden_dim must be greater than 0"
        assert state_dim > 0, "state_dim must be greater than 0"

        self.fc_input_layer = ResLayers(
            hidden_dim,
            stacked_res_hidden_size,
            num_layers,
            hidden_dim,
            bias,
        )

        self.fc_to_mean = nn.Linear(hidden_dim, state_dim, bias)
        self.fc_to_stddev = nn.Linear(hidden_dim, state_dim, bias)

    def forward(self, hidden: torch.Tensor) -> Distribution:
        """
        Args:
            hidden (Tensor): Hidden vectors computed deterministically.

        Returns:
            state (Distribution) : The Normal distribution
        """
        hidden = self.fc_input_layer(hidden)
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
