import os

import gym
from omegaconf import DictConfig
from pynktrombonegym.wrappers.action_by_acceleration import \
    ActionByAcceleration
from pynktrombonegym.wrappers.log1p_mel_spectrogram import Log1pMelSpectrogram

from .array_action import ArrayAction
from .array_voc_state import ArrayVocState
from .normalize_action_range import NormalizeActionRange


def make_env(configs: DictConfig) -> gym.Env:
    """
    Creates an environment instance from a list of audio dir paths.
    
    Args:
    configs: Any: A dictionary of configurations.
    
    Returns:
    gym.Env: The created environment instance.

    """
    
    files = []
    for dataset_dir in configs["dataset_dirs"]:
        for ext in configs["file_exts"]:
            files.extend(
                [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(ext)]
            )

    # ラッパーを順番に適用
    env = Log1pMelSpectrogram(files)
    env = ActionByAcceleration(env, action_scaler=configs["action_scaler"])
    env = NormalizeActionRange(env)
    env = ArrayAction(env, new_step_api=configs["ArrayAction_new_step_api"])
    env = ArrayVocState(env, new_step_api=configs["ArrayVocState_new_step_api"])
    return env
