import os
from typing import Any, List, Optional

import gym
import numpy as np
from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.wrappers import ActionByAcceleration

from .array_action import ArrayAction
from .array_voc_state import ArrayVocState
from .log_mel_spectrogram import LogMelSpectrogram
from .normalize_action_range import NormalizeActionRange


def make_env(
    dataset_dirs: List[Any],
    file_exts: List[str] = [".wav"],
    action_scaler: Optional[float] = None,
    low: float = -1.0,
    high: float = 1.0,
    sample_rate: int = 44100,
    n_mels: int = 80,
    dtype: Any = np.float32,
    default_frequency: float = 400.0,
    log_offset: float = 1e-6,
) -> gym.Env:
    """Creates an wrapped environment instance from a list of audio dir paths.

    Args:
        dataset_dirs (List[Any]): A list of directory paths that contain audio files.
        file_exts (List[str]): A list of file extensions of audio files. Default is [".wav"].
        action_scaler (float): The scaling factor of action. Default is None.
        low (float): The lower limit of action range. Default is -1.0.
        high (float): The upper limit of action range. Default is 1.0.
        sample_rate (int): The sample rate of audio files. Default is 44100.
        n_mels (int): The number of mel bands to generate. Default is 80.
        dtype (Any): The data type of the audio. Default is np.float32.
        default_frequency (float): Default vocal tract frequency.
        log_offset (float): Minimum amplitude of spectrogram to avoid -inf.

    Returns:
        gym.Env: The created environment instance.
    """
    files = create_file_list(dataset_dirs, file_exts)

    base_env = PynkTrombone(
        files,
        sample_rate=sample_rate,
        default_frequency=default_frequency,
    )

    if action_scaler is None:
        action_scaler = base_env.generate_chunk / base_env.sample_rate

    env = LogMelSpectrogram(
        base_env, sample_rate, base_env.stft_window_size, n_mels, log_offset, dtype
    )
    env = ActionByAcceleration(env, action_scaler=action_scaler)
    env = NormalizeActionRange(env, low=low, high=high)
    env = ArrayAction(env)
    env = ArrayVocState(env)

    return env


def create_file_list(dataset_dirs: List[Any], file_exts: List[str]) -> List[str]:
    """Creates a list of audio file paths from A list of directory paths that contain audio files.

    Args:
        dataset_dirs: List[Any]: A list of directory paths that contain audio files.
        file_exts: List[str]: A list of file extensions of audio files.

    Returns:
        List[str]: A list of audio file paths.
    """
    files = []
    for dataset_dir in dataset_dirs:
        if os.path.isdir(dataset_dir):
            for ext in file_exts:
                files.extend(
                    [
                        os.path.join(dataset_dir, f)
                        for f in os.listdir(dataset_dir)
                        if f.endswith(ext)
                    ]
                )
        else:
            raise ValueError(f"{dataset_dir} is not a directory or does not exist.")
    return files
