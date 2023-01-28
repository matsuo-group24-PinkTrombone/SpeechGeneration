import os
from typing import List, Any

import gym
from omegaconf import DictConfig
from pynktrombonegym.wrappers.action_by_acceleration import \
    ActionByAcceleration
from pynktrombonegym.wrappers.log1p_mel_spectrogram import Log1pMelSpectrogram

from .array_action import ArrayAction
from .array_voc_state import ArrayVocState
from .normalize_action_range import NormalizeActionRange


def make_env(dataset_dirs: List[Any], file_exts: List[str], action_scaler) -> gym.Env:
    """
    Creates an environment instance from a list of audio dir paths.
    
    Args:
    configs: Any: A dictionary of configurations.
    
    Returns:
    gym.Env: The created environment instance.

    """
    
    files = []
    for dataset_dir in dataset_dirs:
        for ext in file_exts:
            files.extend(
                [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(ext)]
            )

    # ラッパーを順番に適用
    env = Log1pMelSpectrogram(files)
    env = ActionByAcceleration(env, action_scaler=action_scaler)
    env = NormalizeActionRange(env)
    env = ArrayAction(env)
    env = ArrayVocState(env)
    return env
