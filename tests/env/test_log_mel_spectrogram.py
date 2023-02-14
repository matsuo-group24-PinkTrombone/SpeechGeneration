import glob
from collections import OrderedDict

import numpy as np
from gym import spaces
from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.spaces import ObservationSpaceNames as OSN
from pynktrombonegym.spectrogram import calc_rfft_channel_num

from src.env.log_mel_spectrogram import LogMelSpectrogram

SAMPLE_TARGET_SOUND_FILE_PATHS = glob.glob("data/sample_target_sounds/*")


def test__init__():
    base_env = PynkTrombone(SAMPLE_TARGET_SOUND_FILE_PATHS)
    n_mels = 80
    log_offset = 1e-6
    n_fft = base_env.stft_window_size
    dtype = np.float32
    fft_c = calc_rfft_channel_num(n_fft)
    filter_bank_shape = (n_mels, fft_c)
    wrapper = LogMelSpectrogram(
        base_env, base_env.sample_rate, base_env.stft_window_size, n_mels, log_offset, dtype
    )
    assert wrapper.sample_rate == base_env.sample_rate
    assert wrapper.n_fft == n_fft
    assert wrapper.n_mels == n_mels
    assert wrapper.log_offset == log_offset
    assert wrapper.mel_filter_bank.shape == filter_bank_shape
    assert wrapper.mel_filter_bank.dtype == dtype

    del base_env


def test_define_observation_space():
    base_env = PynkTrombone(SAMPLE_TARGET_SOUND_FILE_PATHS)
    sample_rate = base_env.sample_rate
    n_fft = base_env.stft_window_size
    n_mels = 80
    spect_len = base_env.observation_space[OSN.TARGET_SOUND_SPECTROGRAM].shape[1]
    observation_wrapper = LogMelSpectrogram(base_env, sample_rate, n_fft, n_mels)

    obs_space = observation_wrapper.define_observation_space()

    assert isinstance(obs_space, spaces.Dict)
    assert isinstance(obs_space[OSN.TARGET_SOUND_SPECTROGRAM], spaces.Box)
    assert isinstance(obs_space[OSN.GENERATED_SOUND_SPECTROGRAM], spaces.Box)
    assert obs_space[OSN.TARGET_SOUND_SPECTROGRAM].shape == (n_mels, spect_len)
    assert obs_space[OSN.GENERATED_SOUND_SPECTROGRAM].shape == (n_mels, spect_len)

    del base_env


def test_observation():
    env = PynkTrombone(SAMPLE_TARGET_SOUND_FILE_PATHS)
    sample_rate = env.sample_rate
    n_fft = env.stft_window_size
    n_mels = 80
    log_mel_spectrogram = LogMelSpectrogram(env, sample_rate, n_fft, n_mels)
    shape = log_mel_spectrogram.observation_space[OSN.TARGET_SOUND_SPECTROGRAM].shape

    obs = env.reset()
    obs = log_mel_spectrogram.observation(obs)

    assert isinstance(obs, OrderedDict)
    assert isinstance(obs[OSN.TARGET_SOUND_SPECTROGRAM], np.ndarray)
    assert isinstance(obs[OSN.GENERATED_SOUND_SPECTROGRAM], np.ndarray)
    assert obs[OSN.TARGET_SOUND_SPECTROGRAM].shape == shape
    assert obs[OSN.GENERATED_SOUND_SPECTROGRAM].shape == shape

    del env


def test_log_mel():
    env = PynkTrombone(SAMPLE_TARGET_SOUND_FILE_PATHS)
    sample_rate = env.sample_rate
    n_mels = 80
    n_fft = env.stft_window_size
    log_mel_spectrogram = LogMelSpectrogram(env, sample_rate, n_fft, n_mels)
    target_spectrogram_shape = env.observation_space[OSN.TARGET_SOUND_SPECTROGRAM].shape
    logmel_shape = (n_mels, target_spectrogram_shape[1])
    zero_spectrogram = np.zeros(target_spectrogram_shape)

    zero_logmel = log_mel_spectrogram.log_mel(zero_spectrogram)

    assert isinstance(zero_logmel, np.ndarray)
    assert zero_logmel.shape == logmel_shape
    np.testing.assert_allclose(zero_logmel, np.log(1e-6))

    del env
