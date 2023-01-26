import torch
from src.models.dreamer import Dreamer
from ..datamodules import buffer_names
from src.datamodules.replay_buffer import ReplayBuffer
from src.env.array_voc_state import VocStateObsNames as ObsNames


cls = Dreamer


def test__init__():
    model = cls()