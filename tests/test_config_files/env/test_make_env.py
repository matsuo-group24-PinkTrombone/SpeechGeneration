import gym
import omegaconf
from hydra.utils import instantiate

cfg_file = "configs/env/make_env.yaml"


def test_instantiate():
    cfg = omegaconf.OmegaConf.load(cfg_file)
    obj = instantiate(cfg)

    assert isinstance(obj, gym.Env)
