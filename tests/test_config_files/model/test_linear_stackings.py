import omegaconf
from hydra.utils import instantiate

from src.models.dreamer import Dreamer

cfg_file = "configs/model/linear_stackings.yaml"


def test_instantiate():
    cfg = omegaconf.OmegaConf.load(cfg_file)
    obj = instantiate(cfg)

    assert isinstance(obj, Dreamer)
