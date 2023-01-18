import torch
from torch import Tensor

from src.models.abc import controller as mod


def test_Controller():
    cls = mod.Controller

    try:
        cls()
        assert False, "Not raised TypeError"
    except TypeError:
        pass

    class C(cls):
        def forward(
            self,
            hidden: Tensor,
            state: Tensor,
            target: Tensor,
            controller_hidden: Tensor,
            probabilistic: bool,
        ) -> Tensor:
            return torch.randn(10)

    dummy = torch.randn(10)
    c = C()
    c.forward(dummy, dummy, dummy, dummy, True)
