from typing import Any

import torch.nn as nn
from torch.distributions import Distribution

from ._types import _t_or_any, _tensor_or_any
from .observation_auto_encoder import ObservationDecoder, ObservationEncoder
from .prior import Prior
from .transition import Transition

_dist_or_any = _t_or_any[Distribution]


class World:
    """Abstract world model *interface* class."""

    def __init__(
        self,
        transition: Transition,
        prior: Prior,
        obs_encoder: ObservationEncoder,
        obs_decoder: ObservationDecoder,
        *args: Any,
        **kwds: Any,
    ) -> None:
        """
        Args:
            transition (Transition): Instance of Transition model class.
            prior (Prior): Instance of Prior model class.
            obs_encoder (ObservationEncoder): Instance of Observation Encoder class.
            obs_decoder (ObservationDecoder): Instance of Observation Decoder class.
        """
        super().__init__()

        self.transition = transition
        self.prior = prior
        self.obs_encoder = obs_encoder
        self.obs_decoder = obs_decoder

    def forward(
        self,
        hidden: _tensor_or_any,
        state: _tensor_or_any,
        action: _tensor_or_any,
        next_obs: _tensor_or_any,
        *args: Any,
        **kwds: Any,
    ) -> tuple[_dist_or_any, _dist_or_any, _tensor_or_any]:
        """Make world model transition.

        Args:
            hidden (_tensor_or_any): hidden state `h_t`.
            state (_tensor_or_any): world 'state' `s_t`,
            action (_tensor_or_any): action array data `a_t`,
            next_obs (_tensor_or_any): next step observation data `o_{t+1}`

        Returns:
            next_state_prior (_dist_or_any): Next state by prior. `s^_{t+1}`
            next_state_posterior (_dist_or_any): Next state by Observation Encoder. `s_{t+1}`
            next_hidden (_tensor_or_any): Next hidden state. `h_{t+1}`
        """

        next_hidden = self.transition.forward(hidden, state, action)
        next_state_prior = self.prior.forward(next_hidden)
        next_state_posterior = self.obs_encoder.forward(next_hidden, next_obs)

        return next_state_prior, next_state_posterior, next_hidden

    def eval(self):
        """Set models to evaluation mode."""
        self.transition.eval()
        self.prior.eval()
        self.obs_encoder.eval()
        self.obs_decoder.eval()

    def train(self):
        """Set models to training mode."""
        self.transition.train()
        self.prior.train()
        self.obs_encoder.train()
        self.obs_decoder.train()
