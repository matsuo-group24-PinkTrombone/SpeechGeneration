import torch
from torch import Tensor
from torch.distributions import Normal

from src.models.abc import prior as mod


def test_Prior():
    cls = mod.Prior

    try:
        cls()
        assert False, "Not raised TypeError"
    except TypeError:
        pass

    class C(cls):
        def forward(self, hidden: Tensor) -> Normal:
            return Normal(hidden, hidden.abs())

    dummy = torch.randn(10)
    c = C()
    c.forward(dummy)
