from abc import ABC, abstractmethod

import torch.nn as nn

from ._types import _tensor_or_any as _toa


class Transition(nn.Module, ABC):
    """Abstract transition model class.

    This model gets `state`, `hidden`, and `action`, after returns `next_hidden` in
    :meth:`forward`.
    """

    @abstractmethod
    def forward(self, state: _toa, hidden: _toa, action: _toa) -> _toa:
        pass
