import os

import gym
import omegaconf
import pytest
from hydra.utils import instantiate

cfg_file = "configs/env/make_env.yaml"
cfg = omegaconf.OmegaConf.load(cfg_file)

_MISSING_DATASET_DIR = not all(map(os.path.isdir, cfg.dataset_dirs))


@pytest.mark.skipif(_MISSING_DATASET_DIR, reason="Missing dataset directory.")
def test_instantiate():
    obj = instantiate(cfg)

    assert isinstance(obj, gym.Env)
