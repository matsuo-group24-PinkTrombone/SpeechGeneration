# このファイルは直接実行するため、相対import文を記述しないでください。
import logging
import random
from typing import Optional

import gym
import hydra
import numpy as np
import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf

from src.datamodules.replay_buffer import ReplayBuffer
from src.models.dreamer import Dreamer
from src.trainer import Trainer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Copied from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    """Trains the model."""

    if cfg.get("seed") is not None:
        seed = cfg.get("seed")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        logger.info(f"Reset seed: {seed}")

    logger.info(f"Instantiating env <{cfg.env._target_}>")
    env: gym.Env = hydra.utils.instantiate(cfg.env)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: Dreamer = hydra.utils.instantiate(cfg.model)

    logger.info(f"Instantiating replay_buffer <{cfg.replay_buffer._target_}>")
    replay_buffer = hydra.utils.instantiate(cfg.replay_buffer)
    replay_buffer: ReplayBuffer = replay_buffer(spaces=model.configure_replay_buffer_space(env))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    OmegaConf.resolve(cfg)
    logger.info(f"Training configs:\n{OmegaConf.to_yaml(cfg)}")

    logger.info("Training start!")

    output = trainer.fit(env, replay_buffer, model)

    del env

    logger.info("Training finished.")

    return output


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
