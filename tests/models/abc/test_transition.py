import torch
from torch import Tensor

from src.models.abc import transition as mod


def test_Transition():
    cls = mod.Transition

    try:
        cls()
        assert False, "Not raised TypeError"
    except TypeError:
        pass

    class C(cls):
        def forward(self, state: Tensor, hidden: Tensor, action: Tensor) -> Tensor:
            return torch.randn(10)

    dummy = torch.randn(10)
    c = C()
    c.forward(dummy, dummy, dummy)
