import os
from typing import Any, List, Optional

import gym
import numpy as np
from pynktrombonegym.wrappers import ActionByAcceleration, Log1pMelSpectrogram

from .array_action import ArrayAction
from .array_voc_state import ArrayVocState
from .normalize_action_range import NormalizeActionRange


def make_env(dataset_dirs: List[Any], file_exts: List[str] = ["wav"], 
             action_scaler: Optional[float] = None, low: float = -1.0, high: float = 1.0, 
             sample_rate: int = 44100, n_mels: int = 80, dtype: Any = np.float32) -> gym.Env:
    """
    Creates an environment instance from a list of audio dir paths.

    Args:
        dataset_dirs (List[Any]): A list of directory paths that contain audio files.
        file_exts (Optional[List[str]]): A list of file extensions of audio files. Default is None.
        action_scaler (Optional[float]): The scaling factor of action. Default is None.
        low (Optional[float]): The lower limit of action range. Default is None.
        high (Optional[float]): The upper limit of action range. Default is None.
        sample_rate (int): The sample rate of audio files. Default is 44100.
        n_mels (int): The number of mel bands to generate. Default is 80.
        dtype (Any): The data type of the audio. Default is np.float32.
        
    Returns:
        gym.Env: The created environment instance.
    """
    file_exts = file_exts or [".wav"]
    files = create_file_list(dataset_dirs, file_exts)
    
    base_env_kwargs = {}
    if n_mels is not None:
        base_env_kwargs['n_mels'] = n_mels
    if sample_rate is not None:
        base_env_kwargs['sample_rate'] = sample_rate
    if dtype is not None:
        base_env_kwargs['dtype'] = dtype
    base_env = Log1pMelSpectrogram(files, **base_env_kwargs)
    
    apply_wrappers_kwargs = {}
    if action_scaler is not None:
        apply_wrappers_kwargs['action_scaler'] = action_scaler
    if low is not None:
        apply_wrappers_kwargs['low'] = low
    if high is not None:
        apply_wrappers_kwargs['high'] = high
    env = apply_wrappers(base_env, **apply_wrappers_kwargs)
    return env


def create_file_list(dataset_dirs: List[Any], file_exts: Optional[List[str]] = None) -> List[str]:
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
    if file_exts is None:
        file_exts = [".wav"]
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


def apply_wrappers(env: gym.Env, action_scaler: float = 1.0, low: float = 10.0, high: float = 10.0) -> gym.Env:
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
