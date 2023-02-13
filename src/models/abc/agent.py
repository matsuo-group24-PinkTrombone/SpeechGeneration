from abc import ABC, abstractmethod
from typing import Any

import torch

from ._types import _tensor_or_any
from .controller import Controller
from .observation_auto_encoder import ObservationEncoder
from .transition import Transition


class Agent(ABC):
    """Abstract agent model *interface* class."""

    def __init__(
        self,
        controller: Controller,
        transition: Transition,
        obs_encoder: ObservationEncoder,
        *args: Any,
        **kwds: Any
    ) -> None:
        """
        Args:
            controller (Controller): Instance of Controller model class.
            transition (Transition): Instance of Transition model class.
            obs_encoder (ObservationEncoder): Instance of Observation Encoder model class.
        """

        self.controller = controller
        self.transition = transition
        self.obs_encoder = obs_encoder

        hidden = torch.zeros(1, *transition.hidden_shape)
        controller_hidden = torch.zeros(1, *controller.controller_hidden_shape)

        # Throw device location management to `nn.Module`.
        self.transition.register_buffer("_hidden_of_agent", hidden, False)
        self.controller.register_buffer("_controller_hidden_of_agent", controller_hidden, False)

    @property
    def hidden(self) -> torch.Tensor:
        return self.transition._hidden_of_agent

    @hidden.setter
    def hidden(self, hidden_of_agent: torch.Tensor) -> None:
        self.transition._hidden_of_agent = hidden_of_agent

    @property
    def controller_hidden(self) -> torch.Tensor:
        return self.controller._controller_hidden_of_agent

    @controller_hidden.setter
    def controller_hidden(self, controller_hidden_of_agent: torch.Tensor) -> None:
        self.controller_hidden._controller_hidden_of_agent = controller_hidden_of_agent

    def act(
        self, obs: _tensor_or_any, target: _tensor_or_any, probabilistic: bool
    ) -> _tensor_or_any:
        """Take action using input observation.
        Args:
            obs (_tensor_or_any): Observation from environment.
            target (_tensor_or_any): Target sound data from environment.
            probabilistic (bool): If True, generate action from probability distribution.

        Returns:
            action (_tensor_or_any): Action for stepping environment.
        """
        hidden = self.hidden
        controller_hidden = self.controller_hidden

        state = self.obs_encoder.forward(hidden, obs).sample()
        action, controller_hidden = self.controller.forward(
            hidden, state, target, controller_hidden, probabilistic=probabilistic
        )

        # Update internal hidden state.
        self.hidden = self.transition.forward(hidden, state, action)
        self.controller_hidden = controller_hidden

        return action

    @abstractmethod
    def explore(
        self, obs: _tensor_or_any, target: _tensor_or_any, alpha: _tensor_or_any
    ) -> _tensor_or_any:
        """Take action for exploring environment using input observation.

        Args:
            obs (_tensor_or_any): Observation from environment.
            target (_tensor_or_any): Target sound data from environment.
            alpha (_tensor_or_any): The output of the action model and the mixing ratio of the Gaussian distribution.
        Returns:
            action (_tensor_or_any): Action for exploring environment.
        """
        pass

    def reset(self):
        """Reset internal hidden states."""
        self.hidden.zero_()
        self.controller_hidden.zero_()
