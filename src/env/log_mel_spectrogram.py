import copy
from collections import OrderedDict
from typing import Any

import gym
import librosa
import numpy as np
from gym import spaces
from pynktrombonegym.spaces import ObservationSpaceNames as OSN


class LogMelSpectrogram(gym.ObservationWrapper):
    """Observation Wrapper for converting spectrogram to log mel scale."""

    def __init__(
        self,
        env: gym.Env,
        sample_rate: float,
        n_fft: int,
        n_mels: int = 80,
        log_offset: float = 1e-6,
        dtype: Any = np.float32,
        new_step_api: bool = True,
    ) -> None:
        """
        Args:
            env (gym.Env): PynkTromboneGym Env.
            sample_rate (int): Sampling rate of wave.
            n_fft (int): STFT window size.
            n_mels (int): The size of mel channels.
            log_offset (float): Minimum amplitude of spectrogram to avoid -inf.
            dtype (Any): Mel filter bank data type.
            new_step_api (bool): gym api.
        """
        super().__init__(env, new_step_api)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.log_offset = log_offset
        self.mel_filter_bank = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, dtype=dtype
        )

        self.observation_space = self.define_observation_space()

    observation_space: spaces.Dict

    def define_observation_space(self) -> spaces.Dict:
        """Convert base observation spectrogram to mel spectrogram."""

        base_obs_space = self.env.observation_space
        obs_space = copy.deepcopy(base_obs_space)

        spect_space: spaces.Box = base_obs_space[OSN.TARGET_SOUND_SPECTROGRAM]
        shape = (self.n_mels, spect_space.shape[-1])

        log_mel_space = spaces.Box(-np.inf, np.inf, shape)

        obs_space[OSN.TARGET_SOUND_SPECTROGRAM] = log_mel_space
        obs_space[OSN.GENERATED_SOUND_SPECTROGRAM] = log_mel_space

        return obs_space

    def observation(
        self, observation: OrderedDict[str, np.ndarray]
    ) -> OrderedDict[str, np.ndarray]:
        generated_spect = observation[OSN.GENERATED_SOUND_SPECTROGRAM]
        target_spect = observation[OSN.TARGET_SOUND_SPECTROGRAM]

        generated_log_mel = self.log_mel(generated_spect)
        target_log_mel = self.log_mel(target_spect)

        observation[OSN.GENERATED_SOUND_SPECTROGRAM] = generated_log_mel
        observation[OSN.TARGET_SOUND_SPECTROGRAM] = target_log_mel

        return observation

    def log_mel(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply log mel conversion.
        Args:
            spectrogram (np.ndarray): source spectrogram

        Returns:
            log mel spectrogram (np.ndarray): log mel spectrogram
        """

        return np.log(np.matmul(self.mel_filter_bank, spectrogram) + self.log_offset)
