from abc import ABC, abstractmethod

import torch.nn as nn

from ._types import _tensor_or_any as _toa


class Transition(nn.Module, ABC):
    """Abstract transition model class.

    This model gets `hidden`, `state`, and `action`, after returns `next_hidden` in
    :meth:`forward`.
        i.e. f(h_t, s_t, a_t) -> h_{t+1}
    """

    @abstractmethod
    def forward(self, hidden: _toa, state: _toa, action: _toa) -> _toa:
        """Determistic hidden state (not world 'state'!) transition.
        Args:
            hidden (_tensor_or_any): hidden state h_t.
            state (_tensor_or_any):  world state s_t
            action (_tensor_or_any): action a_t

        Returns:
            next_hidden (_tensor_or_any): next hidden state h_{t+1}
        """
        pass

    @property
    @abstractmethod
    def hidden_shape(self) -> tuple[int]:
        """Returns hidden_state (not world 'state'!) shape.

        Do not contain batch dim.
        """
        pass
