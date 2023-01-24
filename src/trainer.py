import logging

import gym
import torch
from torch.nn.utils import clip_grad_norm_

from .datamodules import buffer_names
from .datamodules.replay_buffer import ReplayBuffer
from .env.array_voc_state import VocStateObsNames as ObsNames
from .models.dreamer import Dreamer

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for Dreamer."""

    def __init__(
        self,
        num_episode: int = 1,
        collect_experience_interval: int = 100,
        batch_size: int = 8,
        chunk_size: int = 64,
        gradient_clip_value: float = 100.0,
    ) -> None:
        """
        Args:

        """
        self.__dict__.update(locals())  # Add all input args to class attribute.

    def fit(self, env: gym.Env, replay_buffer: ReplayBuffer, model: Dreamer) -> None:
        """Fit
        Args:

        """

        world_optimizer, controller_optimizer = model.configure_optimizers()

        current_step = 0

        logger.info("Fit started.")
        for episode in range(self.num_episode):
            logger.info(f"Episode {episode} is started.")

            logger.debug("Collecting experiences...")
            replay_buffer = model.collect_experiences(
                env, replay_buffer, self.num_collect_experience_steps
            )
            logger.debug("Collected experiences.")

            for collect_interval in range(self.collect_experience_interval):
                logger.debug(f"Collect interval: {collect_interval}")

                # Training World Model.
                experiences_dict = replay_buffer.sample(
                    self.batch_size, self.chunk_size, chunk_first=True
                )
                loss_dict, experiences_dict = model.world_training_step(experiences_dict)

                loss: torch.Tensor = loss_dict["loss"]
                world_optimizer.zero_grad()
                loss.backward()
                params = []
                for p in world_optimizer.param_groups:
                    params += p["params"]
                clip_grad_norm_(params, self.gradient_clip_value)
                world_optimizer.step()

                # -- logging --

                # Training Controller model.
