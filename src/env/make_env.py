import os
from typing import Any, List

import gym
from omegaconf import DictConfig
from pynktrombonegym.wrappers.action_by_acceleration import \
    ActionByAcceleration
from pynktrombonegym.wrappers.log1p_mel_spectrogram import Log1pMelSpectrogram

from .array_action import ArrayAction
from .array_voc_state import ArrayVocState
from .normalize_action_range import NormalizeActionRange


def make_env(dataset_dirs: List[Any], file_exts: List[str], action_scaler: float, low: float, high: float) -> gym.Env:
    """
    Creates an environment instance from a list of audio dir paths.

    Args:
        dataset_dirs: List[Any]: A list of directory paths that contain audio files.
        file_exts: List[str]: A list of file extensions of audio files.
        action_scaler: float: The scaling factor of action.
        low: float: The lower limit of action range.
        high: float: The upper limit of action range.
        Returns:
        gym.Env: The created environment instance.
    """
    files = create_file_list(dataset_dirs, file_exts)
    env = Log1pMelSpectrogram(files)
    env = apply_wrappers(env, action_scaler, low, high)
    return env


def create_file_list(dataset_dirs: List[Any], file_exts: List[str]) -> List[str]:
    """
    Creates a list of audio file paths.

    Args:
        dataset_dirs: List[Any]: A list of directory paths that contain audio files.
        file_exts: List[str]: A list of file extensions of audio files.

    Returns:
        List[str]: A list of audio file paths.
    """
    files = []
    if not dataset_dirs:
        return files
    file_exts = [ext.lower() for ext in file_exts]
    for dataset_dir in dataset_dirs:
        if os.path.isdir(dataset_dir):
            for ext in file_exts:
                files.extend(
                    [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(ext)]
                )
        else:
            print(f"{dataset_dir} is not a directory.")
    return files


def apply_wrappers(env: gym.Env, action_scaler: float, low: float, high: float) -> gym.Env:
    """
    Apply wrappers to the environment.

    Args:
        env: gym.Env: The environment to apply wrappers.
        action_scaler: float: The scaling factor of action.
        low: float: The lower limit of action range.
        high: float: The upper limit of action range.

    Returns:
        gym.Env: The environment with applied wrappers.
    """
    env = ActionByAcceleration(env, action_scaler=action_scaler)
    env = NormalizeActionRange(env, low=low, high=high)
    env = ArrayAction(env)
    env = ArrayVocState(env)
    return env
