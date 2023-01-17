from abc import ABC, abstractmethod

import torch.nn as nn
from torch.distributions import Distribution

from ._types import _t_or_any, _tensor_or_any


class ObservationEncoder(nn.Module, ABC):
    """Abstract observation encoder model class.

    This model gets `hidden` and `obs` and returns `state` in :meth:`forward`     i.e. s_t ~
    q_E(s_t|h_t, o_t)
    """

    @abstractmethod
    def embed_observation(self, obs: _tensor_or_any) -> _tensor_or_any:
        """Embed observation to latent space.

        Args:
            obs (_tensor_or_any): observation data.
                For instance, obs=(voc state v_t, generated sound g_t) <- Tuple of Tensor!
        Returns:
            embedded_obs (_tensor_or_any): Embedded observation.
        """
        pass

    @abstractmethod
    def encode(
        self, hidden: _tensor_or_any, embedded_obs: _tensor_or_any
    ) -> _t_or_any[Distribution]:
        """Encode hidden state and embedded_obs to world 'state'.
        Args:
            hidden (_tensor_or_any): hidden state `h_t`
            embedded_obs (_tensor_or_any): Embedded obsevation data.

        Returns:
            state (_t_or_any[Distribution]): world state `s_t`.
                Type is `Distiribution` class for computing kl divergence.
        """
        pass

    @property
    @abstractmethod
    def state_shape(self) -> tuple[int]:
        """Returns world 'state' shape.

        Do not contain batch dim.
        """

    def forward(self, hidden: _tensor_or_any, obs: _tensor_or_any) -> _t_or_any[Distribution]:
        """Forward path of encoder."""
        embedded_obs = self.embed_observation(obs)
        state = self.encode(hidden, embedded_obs)
        return state
