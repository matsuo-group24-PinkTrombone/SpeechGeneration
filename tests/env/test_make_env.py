
import glob

import gym
from omegaconf import OmegaConf

from src.env.make_env import make_env


def test_make_env():
    dataset_dirs = "data/*"
    dict_conf = {
        "dataset_dirs": glob.glob(dataset_dirs),
        "file_exts": [".wav"]
    }
    conf = OmegaConf.create(dict_conf)
    env = make_env(conf)
    assert isinstance(env, gym.Env)