import os

import gym
from array_action import ArrayAction
from array_voc_state import ArrayVocState
from normalize_action_range import NormalizeActionRange
from omegaconf import DictConfig
from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.wrappers.action_by_acceleration import \
    ActionByAcceleration
from pynktrombonegym.wrappers.log1p_mel_spectrogram import Log1pMelSpectrogram


def make_env(configs: DictConfig) -> gym.Env:
    """
    Creates an environment instance from a list of audio dir paths.
    
    Args:
    configs: Any: A dictionary of configurations, including 'dataset_dirs' and 'file_exts'.
    
    Returns:
    gym.Env: The created environment instance.

    """
    
    files = []
    for dataset_dir in configs["dataset_dirs"]:
        for ext in configs["file_exts"]:
            files.extend(
                [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(ext)]
            )
    env = PynkTrombone(files)

    # ラッパーを順番に適用
    env = Log1pMelSpectrogram(env)
    env = ActionByAcceleration(env)
    env = NormalizeActionRange(env)
    env = ArrayAction(env)
    env = ArrayVocState(env)
    return env
