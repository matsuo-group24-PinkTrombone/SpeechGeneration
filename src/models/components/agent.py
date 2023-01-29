from typing import Any

import torch

from ..abc.agent import Agent as AbstractAgent
from .controller import Controller
from .observation_auto_encoder import ObservationEncoder
from .transition import Transition

class Agent(AbstractAgent):
    def __init__(
        self,
        controller: Controller,
        transition: Transition,
        obs_encoder: ObservationEncoder,
        *args: Any,
        **kwds: Any
        ):
        """
        Args:
            controller (Controller): Instance of Controller model class.
            transition (Transition): Instance of Transition model class.
            obs_encoder (ObservationEncoder): Instance of Observation Encoder model class.
        """
        super().__init__(
            controller=controller,
            transition=transition,
            obs_encoder=obs_encoder,
            *args,
            **kwds
        )
        
    def explore(self, obs: torch.Tensor, target: torch.Tensor, alpha: int) -> torch.Tensor:
        """Take action for exploring environment using input observation.

        Args:
            obs (_tensor_or_any): Observation from environment.
            target (_tensor_or_any): Target sound data from environment.
            alpha (_tensor_or_any): The output of the action model and the mixing ratio of the Gaussian distribution.
        Returns:
            action (_tensor_or_any): Action for exploring environment.
        """
        state = self.obs_encoder.forward(self.hidden, obs).sample()
        action, controller_hidden = self.controller.forward(
            self.hidden, state, target, self.controller_hidden, probabilistic=probabilistic
        )

        # Update internal hidden state.
        self.hidden = self.transition.forward(self.hidden, state, action)
        self.controller_hidden = controller_hidden

        batch_size, action_size = action.size()
        noise = torch.rand(batch_size,action_size)
        return action * alpha + (1 - alpha) * noise