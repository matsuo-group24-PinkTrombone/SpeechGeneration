from typing import Any

import torch

from ..abc.agent import Agent as AbstractAgent
from ..abc.controller import Controller
from ..abc.observation_auto_encoder import ObservationEncoder
from ..abc.transition import Transition


class Agent(AbstractAgent):
    def __init__(
        self,
        controller: Controller,
        transition: Transition,
        obs_encoder: ObservationEncoder,
        action_noise_ratio: float = 0.5,
    ):
        """
        Args:
            controller (Controller): Instance of Controller model class.
            transition (Transition): Instance of Transition model class.
            obs_encoder (ObservationEncoder): Instance of Observation Encoder model class.
            action_noise_ratio (float): Mixing ratio of action noise. Max 1, Min 0. If 1, action is only noise.
        """
        assert 0.0 <= action_noise_ratio <= 1.0

        super().__init__(
            controller=controller,
            transition=transition,
            obs_encoder=obs_encoder,
        )

        self.action_noise_ratio = action_noise_ratio

    def explore(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Take action for exploring environment using input observation.

        Args:
            obs (_tensor_or_any): Observation from environment.
            target (_tensor_or_any): Target sound data from environment.
            alpha (_tensor_or_any): The output of the action model and the mixing ratio of the Gaussian distribution.
        Returns:
            action (_tensor_or_any): Action for exploring environment.
        """
        action = self.act(obs, target, True)

        noise = torch.rand_like(action) * 2 - 1
        action = noise * self.action_noise_ratio + (1 - self.action_noise_ratio) * action

        return action
