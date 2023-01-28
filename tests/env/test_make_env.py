
import glob

import gym
from omegaconf import OmegaConf

from src.env.make_env import make_env

sample_target_sound_dir_paths = glob.glob("data/**/", recursive=False)
configs = OmegaConf.create(
        {
            "dataset_dirs": sample_target_sound_dir_paths,
            "file_exts": [".wav"],
            "action_scaler": 1.0,
        }
    )

def test__init__():
    env = make_env(**configs)
    assert isinstance(env, gym.Env)
