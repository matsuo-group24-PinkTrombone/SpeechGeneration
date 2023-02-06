import hydra
from omegaconf import OmegaConf

from src.datamodules.replay_buffer import ReplayBuffer
from src.env.make_env import make_env
from src.models.dreamer import Dreamer
from tests.models.test_dreamer import args

cfg_file = "configs/replay_buffer/replay_buffer.yaml"
dreamer = Dreamer(*args)


def test_instantiate():
    env = make_env(["data/sample_target_sounds"])
    cfg = OmegaConf.load(cfg_file)
    obj = hydra.utils.instantiate(cfg)
    obj = obj(spaces=dreamer.configure_replay_buffer_space(env))

    assert isinstance(obj, ReplayBuffer)

    del env
