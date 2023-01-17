import torch
from torch import Tensor
from torch.distributions import Normal

from src.models.abc import observation_auto_encoder as mod


def test_ObservationEncoder():
    cls = mod.ObservationEncoder

    try:
        cls()
        assert False, "Not raised TypeError"
    except TypeError:
        pass

    class C(cls):
        def embed_observation(self, obs: Tensor) -> Tensor:
            return obs + 1

        def encode(self, hidden: Tensor, embedded_obs: Tensor) -> Normal:
            return Normal(hidden, embedded_obs)

        @property
        def state_shape(self) -> tuple[int]:
            return (10,)

        def forward(self, hidden: Tensor, obs: Tensor) -> Normal:
            return super().forward(hidden, obs)

    dummy = torch.randn(10)
    c = C()
    c.embed_observation(dummy)
    c.encode(dummy, dummy.abs())
    c.state_shape
    norm = c.forward(dummy, dummy.abs())
    assert torch.all(norm.loc == dummy)
    assert torch.all(norm.scale == dummy.abs() + 1)
