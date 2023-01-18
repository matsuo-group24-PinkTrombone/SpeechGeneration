from abc import ABC, abstractmethod

import torch.nn as nn
from torch.distributions import Distribution

from ._types import _t_or_any, _tensor_or_any


class Prior(nn.Module, ABC):
    """Abstract prior model class.

    This model gets `hidden` and returns `state` in :meth:`forward`

        i.e. s_t ~ p_s(s_t | h_t)
    """

    @abstractmethod
    def forward(self, hidden: _tensor_or_any) -> _t_or_any[Distribution]:
        """Convert hidden state to world 'state'.

        Args:
            hidden (_tensor_or_any): hidden state `h_t`.

        Returns:
            state (_t_or_any[Distribution]): prior world state `s_t`.
                Type is `Distribution` class for computing kl divergence.
        """
        pass
