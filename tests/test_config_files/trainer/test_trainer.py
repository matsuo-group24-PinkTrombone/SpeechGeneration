import omegaconf
from hydra.utils import instantiate

from src.trainer import Trainer

cfg_file = "configs/trainer/trainer.yaml"


def test_instantiate():
    cfg = omegaconf.OmegaConf.load(cfg_file)
    cfg.checkpoint_destination_path = "logs/test_trainer/checkpoints"
    cfg.tensorboard.log_dir = "logs/test_trainer/tensorboard"
    obj = instantiate(cfg)

    assert isinstance(obj, Trainer)
