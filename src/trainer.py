import logging
from collections import OrderedDict
from typing import Any, Optional
from datetime import datetime
import os

import gym
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from .datamodules import buffer_names
from .datamodules.replay_buffer import ReplayBuffer
from .env.array_voc_state import VocStateObsNames as ObsNames
from .models.dreamer import Dreamer

logger = logging.getLogger(__name__)


class CheckPointNames:
    MODEL = "model"
    WORLD_OPTIMIZER = "world_optimizer"
    CONTROLLER_OPTIMIZER = "controller_optimizer"


class Trainer:
    """Trainer class for Dreamer."""

    def __init__(
        self,
        log_dir: str,
        num_episode: int = 1,
        collect_experience_interval: int = 100,
        batch_size: int = 8,
        chunk_size: int = 64,
        gradient_clip_value: float = 100.0,
        evaluation_interval=10,
        model_save_interval=20,
        checkpoint_path: Optional[Any] = None,
        device: Any = "cpu",
        dtype: Any = torch.float32,
    ) -> None:
        """
        Args:

        """
        self.__dict__.update(locals())  # Add all input args to class attribute.

    def fit(self, env: gym.Env, replay_buffer: ReplayBuffer, model: Dreamer) -> None:
        """Fit
        Args:

        """

        self.setup_model_attribute(model)

        model = model.to(self.device, self.dtype)

        world_optimizer, controller_optimizer = model.configure_optimizers()
        if self.checkpoint_path is not None:
            self.load_checkpoint(model, world_optimizer, controller_optimizer)

        current_step = 0

        logger.info("Fit started.")
        for episode in range(self.num_episode):
            logger.info(f"Episode {episode} is started.")

            model.current_episode = episode
            model.current_step = current_step

            logger.debug("Collecting experiences...")
            model.collect_experiences(env, replay_buffer)
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

                # ---- Training Controller model. -----
                loss_dict, experiences_dict = model.controller_training_step(experiences_dict)

                loss: torch.Tensor = loss_dict["loss"]
                controller_optimizer.zero_grad()
                loss.backward()
                params = []
                for p in controller_optimizer.param_groups:
                    params += p["params"]
                clip_grad_norm_(params, self.gradient_clip_value)
                controller_optimizer.step()

                # logging

                if current_step % self.evaluation_interval == 0:
                    # ----- Evaluation steps -----
                    loss_dict = model.evaluation_step(env)

                    # logging

                if current_step % self.model_save_interval == 0:
                    if self.checkpoint_path is None:
                        save_path = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.ckpt"
                    else:
                        save_path = os.path.join(self.load_checkpoint, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.ckpt")        
                    self.save_checkpoint(save_path, model, world_optimizer, controller_optimizer)

                current_step += 1

        if self.checkpoint_path is None:
            save_path = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.ckpt"
        else:
            save_path = os.path.join(self.load_checkpoint, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.ckpt")
        self.save_checkpoint(save_path, model, world_optimizer, controller_optimizer)

    def setup_model_attribute(self, model: Dreamer):
        """Add attribute for model training.

        Call this begin of training.
        Args:
            model (Dreamer): Dreamer model class.
        """
        model.device = torch.device(self.device)
        model.dtype = self.dtype
        model.current_episode = 0
        model.current_step = 0
        model.tensorboard = SummaryWriter(log_dir=self.log_dir)

    def save_checkpoint(
        self, path: Any, model: Dreamer, world_optim: Optimizer, controller_optim: Optimizer
    ) -> None:
        """Saving checkpoint."""
        ckpt = OrderedDict()
        ckpt[CheckPointNames.MODEL] = model.state_dict()
        ckpt[CheckPointNames.WORLD_OPTIMIZER] = world_optim.state_dict()
        ckpt[CheckPointNames.CONTROLLER_OPTIMIZER] = controller_optim.state_dict()

        torch.save(ckpt, path)

    def load_checkpoint(
        self, path: Any, model: Dreamer, world_optim: Optimizer, controller_optim: Optimizer
    ):
        ckpt = torch.load(path, self.device)
        model.load_state_dict(ckpt[CheckPointNames.MODEL])
        world_optim.load_state_dict(ckpt[CheckPointNames.WORLD_OPTIMIZER])
        controller_optim.load_state_dict(ckpt[CheckPointNames.CONTROLLER_OPTIMIZER])
