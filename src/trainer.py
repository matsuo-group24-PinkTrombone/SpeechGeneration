import logging
import os
from collections import OrderedDict
from datetime import datetime
from pprint import pformat
from typing import Any, Optional

import gym
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        checkpoint_destination_path: str,
        tensorboard: SummaryWriter,
        num_episode: int = 1,
        collect_experience_interval: int = 100,
        batch_size: int = 8,
        chunk_size: int = 64,
        gradient_clip_value: float = 100.0,
        evaluation_interval=10,
        model_save_interval=20,
        saved_checkpoint_path: Optional[Any] = None,
        console_log_every_n_step: int = 1,
        log_every_n_steps: int = 1,
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
        if self.saved_checkpoint_path is not None:
            self.load_checkpoint(
                self.saved_checkpoint_path, model, world_optimizer, controller_optimizer
            )

        current_step = 0

        logger.info("Fit started.")
        for episode in tqdm(range(self.num_episode)):
            logger.info(f"Episode {episode} is started.")

            model.current_episode = episode

            logger.info("Collecting experiences...")
            model.collect_experiences(env, replay_buffer)
            logger.info("Collected experiences.")

            for collect_interval in tqdm(range(self.collect_experience_interval)):
                model.current_step = current_step
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
                if current_step % self.console_log_every_n_step == 0:
                    log_loss = pformat(loss_dict)
                    logger.info(log_loss)

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
                if current_step % self.console_log_every_n_step == 0:
                    log_loss = pformat(loss_dict)
                    logger.info(log_loss)

                if current_step % self.evaluation_interval == 0:
                    # ----- Evaluation steps -----
                    loss_dict = model.evaluation_step(env)

                    # logging
                    log_loss = pformat(loss_dict)
                    logger.info(log_loss)
                if current_step % self.model_save_interval == 0:
                    file_name = f"episode{episode}_step{current_step}.ckpt"
                    save_path = os.path.join(
                        self.checkpoint_destination_path,
                        file_name,
                    )
                    self.save_checkpoint(save_path, model, world_optimizer, controller_optimizer)

                current_step += 1

        file_name = f"episode{episode}_step{current_step}.ckpt"
        save_path = os.path.join(self.checkpoint_destination_path, file_name)
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
        model.tensorboard = self.tensorboard
        model.log_every_n_steps = self.log_every_n_steps

    def save_checkpoint(
        self, path: Any, model: Dreamer, world_optim: Optimizer, controller_optim: Optimizer
    ) -> None:
        """Saving checkpoint."""
        ckpt = OrderedDict()
        ckpt[CheckPointNames.MODEL] = model.state_dict()
        ckpt[CheckPointNames.WORLD_OPTIMIZER] = world_optim.state_dict()
        ckpt[CheckPointNames.CONTROLLER_OPTIMIZER] = controller_optim.state_dict()

        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(
        self, path: Any, model: Dreamer, world_optim: Optimizer, controller_optim: Optimizer
    ) -> None:
        """Load checkpoint."""
        ckpt = torch.load(path, self.device)
        model.load_state_dict(ckpt[CheckPointNames.MODEL])
        world_optim.load_state_dict(ckpt[CheckPointNames.WORLD_OPTIMIZER])
        controller_optim.load_state_dict(ckpt[CheckPointNames.CONTROLLER_OPTIMIZER])
        logger.info(f"Loaded checkpoint from {path}")
