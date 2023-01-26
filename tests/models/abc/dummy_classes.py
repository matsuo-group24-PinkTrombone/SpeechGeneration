from typing import Any

import torch
from torch.nn import Linear
from torch import Tensor
from torch.distributions import Normal

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

    def forward(self, hidden: Tensor, state: Tensor, action: Tensor) -> Tensor:
        return torch.randn(hidden.shape)

    @property
    def hidden_shape(self) -> tuple[int]:
        return self._hidden_shape


class DummyPrior(Prior):
    """Dummy prior model class for testing."""

    def __init__(self, state_shape: tuple[int], *args: Any, **kwds: Any) -> None:
        super().__init__()
        self._state_shape = state_shape
        self.dmy_lyr = Linear(8, 16)

    def forward(self, hidden: Tensor) -> Normal:
        shape = (hidden.size(0), *self.state_shape)
        mean = torch.zeros(shape).type_as(hidden)
        std = torch.ones_like(mean)
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

    def embed_observation(self, obs: tuple[Tensor, Tensor]) -> Tensor:
        v, g = obs
        shape = (v.size(0), *self.embedded_obs_shape)
        return torch.randn(shape).type_as(v)

    def encode(self, hidden: Tensor, embedded_obs: Tensor) -> Normal:
        shape = (hidden.size(0), *self.state_shape)
        mean = torch.zeros(shape).type_as(hidden)
        std = torch.ones_like(mean)
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

    def forward(self, hidden: Tensor, state: Tensor) -> Tensor:
        vs_shape = (hidden.size(0), *self._voc_state_shape)
        gs_shape = (hidden.size(0), *self._generated_sound_shape)
        return torch.randn(vs_shape).type_as(hidden), torch.randn(gs_shape).type_as(hidden)


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

    def forward(
        self,
        hidden: Tensor,
        state: Tensor,
        target: Tensor,
        controller_hidden: Tensor,
        probabilistic: bool,
    ) -> tuple[Tensor, Tensor]:
        shape = (hidden.size(0), *self._action_shape)
        return torch.rand(shape) * 2 + 0.5, torch.randn_like(controller_hidden)

    @property
    def controller_hidden_shape(self) -> tuple[int]:
        return self._controller_hidden_shape


class DummyWorld(World):
    """Dummy world model interface class for testing."""

    pass


class DummyAgent(Agent):
    """Dummy agent model interface class for testing."""

    def explore(self, obs: tuple[Tensor, Tensor], target: Tensor) -> Tensor:
        action = self.act(obs, target, True)
        return action
