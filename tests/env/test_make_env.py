
import glob

import gym
from omegaconf import OmegaConf

from src.env.make_env import make_env

sample_target_sound_dir_paths = glob.glob("data/**/", recursive=False)

def test_make_env():
    configs = OmegaConf.create(
        {
            "dataset_dirs": sample_target_sound_dir_paths,
            "file_exts": [".wav"],
            "action_scaler": 1.0,
            "ArrayAction_new_step_api": True,
            "ArrayVocState_new_step_api": True,
        }
    )
    env = make_env(configs)
    assert isinstance(env, gym.Env)
