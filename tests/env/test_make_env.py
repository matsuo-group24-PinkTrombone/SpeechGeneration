
import glob
import os
import tempfile

import gym
from omegaconf import OmegaConf
from pynktrombonegym.wrappers.log1p_mel_spectrogram import Log1pMelSpectrogram

from src.env.make_env import apply_wrappers, create_file_list, make_env

sample_target_sound_file_paths = glob.glob("data/sample_target_sounds/*")
sample_target_sound_dir_paths = list(glob.glob("data/**/", recursive=True))


def test__init__():
    configs = OmegaConf.create(
        {
            "dataset_dirs": sample_target_sound_dir_paths,
            "file_exts": [".wav"],
            "action_scaler": 1.0,
            "low": -10.0,
            "high": 10.0,
        }
    )
    env = make_env(**configs)
    assert isinstance(env, gym.Env)


def test_create_file_list():
    # create temporary directories
    temp_dir1 = tempfile.TemporaryDirectory()
    temp_dir2 = tempfile.TemporaryDirectory()
    
    # create temporary files
    temp_file1 = tempfile.NamedTemporaryFile(dir=temp_dir1.name, suffix='.wav')
    temp_file2 = tempfile.NamedTemporaryFile(dir=temp_dir1.name, suffix='.flac')
    temp_file3 = tempfile.NamedTemporaryFile(dir=temp_dir2.name, suffix='.txt')
    
    # test1: case of directory exist
    dataset_dirs = [temp_dir1.name, temp_dir2.name]
    file_exts = [".wav", ".flac"]
    expected_output = [temp_file1.name, temp_file2.name]
    assert create_file_list(dataset_dirs, file_exts) == expected_output
    
    # test2: case of directory not exist
    dataset_dirs = [temp_dir1.name, "not_exist_dir"]
    file_exts = [".wav", ".flac"]
    expected_output = [temp_file1.name, temp_file2.name]
    assert create_file_list(dataset_dirs, file_exts) == expected_output # not_exist_dir is not a directory.

    # test3: case of empty input
    dataset_dirs = []
    file_exts = [".wav", ".flac"]
    expected_output = []
    assert create_file_list(dataset_dirs, file_exts) == expected_output

    # test4: case of empty output 
    dataset_dirs = [temp_dir1.name]
    file_exts = [".mp3"]
    expected_output = []
    assert create_file_list(dataset_dirs, file_exts) == expected_output
    
    # clean up temporary directories
    temp_dir1.cleanup()
    temp_dir2.cleanup()
    
    # test for sample_target_sound_dir_paths
    assert create_file_list(sample_target_sound_dir_paths, [".wav"]) == glob.glob("data/**/*.wav", recursive=True)


def test_apply_wrappers():
    base_env = Log1pMelSpectrogram(sample_target_sound_file_paths)
    env = apply_wrappers(base_env)
    assert isinstance(env, gym.Env)