from abc import ABC, abstractmethod

import torch.nn as nn

from ._types import _tensor_or_any


class Controller(nn.Module, ABC):
    r"""Abstract controller model class.

    This model gets `hidden`, `state`, `target` and `controller_hidden`,
    after returns `action` and `controller_hidden` in :meth:`forward`.

        i.e. C(a_t | h_t, s_t, \tau_t, hc_t}) -> a_t, hc_{t+1}

    Note: action range is must be [-1, 1] for action normalization.
    """

    @abstractmethod
    def forward(
        self,
        hidden: _tensor_or_any,
        state: _tensor_or_any,
        target: _tensor_or_any,
        controller_hidden: _tensor_or_any,
        probabilistic: bool,
    ) -> tuple[_tensor_or_any, _tensor_or_any]:
        r"""Take action. If `probabilistic` is True, sampling from normal distribution.

        Args:
            hidden (_tensor_or_any): hidden state `h_t`.
            state (_tensor_or_any): world state `s_t`.
            target (_tensor_or_any): target sound `\tau_t`
            controller_hidden (_tensor_or_any): controller hidden state `hc_t`.
            probabilistic (bool): If True, sample action from normal distribution.
        Returns:
            action (_tensor_or_any): action data `a_t`. The value range must be [-1, 1].
            next_controller_hidden (_tensor_or_any): Next controller hidden state `hc_{t+1}`.
        """
        pass

    @property
    def controller_hidden_shape(self) -> tuple[int]:
        """Returns controller hidden state shape.

        Do not contain batch dim.
        """
        pass
