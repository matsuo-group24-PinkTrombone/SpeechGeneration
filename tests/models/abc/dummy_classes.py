from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Linear

from src.models.abc.agent import Agent
from src.models.abc.controller import Controller
from src.models.abc.observation_auto_encoder import (
    ObservationDecoder,
    ObservationEncoder,
)
from src.models.abc.prior import Prior
from src.models.abc.transition import Transition
from src.models.abc.world import World


class DummyTransition(Transition):
    """Dummy transition model class for testing."""

    def __init__(self, hidden_shape: tuple[int], *args: Any, **kwds: Any) -> None:
        super().__init__()
        self._hidden_shape = hidden_shape
        self.dmy_lyr = Linear(8, 16)
        self.dmy_w = nn.Parameter(torch.zeros(8))

    def forward(self, hidden: Tensor, state: Tensor, action: Tensor) -> Tensor:
        next_hidden = torch.randn_like(hidden)
        return hidden + (next_hidden + state.mean() + action.mean()) * self.dmy_w.mean()

    @property
    def hidden_shape(self) -> tuple[int]:
        return self._hidden_shape


class DummyPrior(Prior):
    """Dummy prior model class for testing."""

    def __init__(self, state_shape: tuple[int], *args: Any, **kwds: Any) -> None:
        super().__init__()
        self._state_shape = state_shape
        self.dmy_lyr = Linear(8, 16)
        self.dmy_w = nn.Parameter(torch.zeros(8))

    def forward(self, hidden: Tensor) -> Normal:
        shape = (hidden.size(0), *self.state_shape)
        mean = (torch.zeros(shape).type_as(hidden) + hidden.mean()) * self.dmy_w.mean()
        std = torch.ones_like(mean) + mean.abs()

        return Normal(mean, std)

    @property
    def state_shape(self) -> tuple[int]:
        return self._state_shape


class DummyObservationEncoder(ObservationEncoder):
    """Dummy observation encoder model class for testing."""

    def __init__(
        self, state_shape: tuple[int], embedded_obs_shape: tuple[int], *args: Any, **kwds: Any
    ) -> None:
        super().__init__()
        self._state_shape = state_shape
        self.embedded_obs_shape = embedded_obs_shape
        self.dmy_lyr = Linear(8, 16)
        self.dmy_w = nn.Parameter(torch.zeros(8))

    def embed_observation(self, obs: tuple[Tensor, Tensor]) -> Tensor:
        v, g = obs
        shape = (v.size(0), *self.embedded_obs_shape)
        emb = torch.randn(shape, requires_grad=True).type_as(v)
        return emb + (v.mean() + g.mean()) * self.dmy_w.mean()

    def encode(self, hidden: Tensor, embedded_obs: Tensor) -> Normal:
        shape = (hidden.size(0), *self.state_shape)
        mean = (
            torch.zeros(shape).type_as(self.dmy_w)
            + (hidden.mean() + embedded_obs.mean()) * self.dmy_w.mean()
        )
        std = torch.ones_like(mean) + mean.abs()

        return Normal(mean, std)

    @property
    def state_shape(self) -> tuple[int]:
        return self._state_shape


class DummyObservationDecoder(ObservationDecoder):
    """Dummy observation decoder model class for testing."""

    def __init__(
        self,
        voc_state_shape: tuple[int],
        generated_sound_shape: tuple[int],
        *args: Any,
        **kwds: Any
    ) -> None:
        super().__init__()
        self._voc_state_shape = voc_state_shape
        self._generated_sound_shape = generated_sound_shape
        self.dmy_lyr = Linear(8, 16)
        self.dmy_w = nn.Parameter(torch.zeros(8))

    def forward(self, hidden: Tensor, state: Tensor) -> Tensor:
        vs_shape = (hidden.size(0), *self._voc_state_shape)
        gs_shape = (hidden.size(0), *self._generated_sound_shape)
        vs, gs = torch.randn(vs_shape, requires_grad=True).type_as(hidden), torch.randn(
            gs_shape, requires_grad=True
        ).type_as(hidden)

        vs = vs + (hidden.mean() + state.mean()) * self.dmy_w.mean()
        gs = gs + (hidden.mean() + state.mean()) * self.dmy_w.mean()

        return vs, gs


class DummyController(Controller):
    """Dummy controller model class for testing."""

    def __init__(
        self,
        action_shape: tuple[int],
        controller_hidden_shape: tuple[int],
        *args: Any,
        **kwds: Any
    ) -> None:
        super().__init__()

        self._action_shape = action_shape
        self._controller_hidden_shape = controller_hidden_shape
        self.dmy_lyr = Linear(8, 16)
        self.dmy_w = nn.Parameter(torch.zeros(8))

    def forward(
        self,
        hidden: Tensor,
        state: Tensor,
        target: Tensor,
        controller_hidden: Tensor,
        probabilistic: bool,
    ) -> tuple[Tensor, Tensor]:
        shape = (hidden.size(0), *self._action_shape)
        act, h_c = torch.rand(shape, requires_grad=True) * 2 - 1, torch.randn_like(
            controller_hidden, requires_grad=True
        )
        act, h_c = act.type_as(self.dmy_w), h_c.type_as(self.dmy_w)
        # act = act + (hidden.mean() + state.mean() + target.mean() + controller_hidden.mean() + self.dmy_w.mean()) / 1e+10
        # act = act / act.abs().max()
        h_c = (
            h_c
            + (hidden.mean() + state.mean() + target.mean() + controller_hidden.mean())
            * self.dmy_w.mean()
        )
        return act, h_c


    @property
    def controller_hidden_shape(self) -> tuple[int]:
        return self._controller_hidden_shape


class DummyWorld(World):
    """Dummy world model interface class for testing."""

    pass


class DummyAgent(Agent):
    """Dummy agent model interface class for testing."""

    def __init__(self, action_shape: tuple[int], *args, **kwds):
        super().__init__(*args, **kwds)
        self.action_shape = action_shape

    def explore(self, obs: tuple[Tensor, Tensor], target: Tensor) -> Tensor:
        w = obs[0].mean() + obs[1].mean() + target.mean()
        action = torch.rand(self.action_shape, requires_grad=True).type_as(w) + w * 0.0
        action = action / action.abs().max()
        return action
