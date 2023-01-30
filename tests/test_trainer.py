from functools import partial

import pytest
import torch
from torch.optim import SGD

from src.models.abc.world import World
from src.models.components.conformer_decoder_fastspeech2 import ConformerDecoder as CD
from src.models.components.controller import Controller as Ctrl
from src.models.components.posterior_encoder_vits import PosteriorEncoderVITS as PE
from src.models.components.prior_encoder import Prior
from src.models.components.transition import Transition
from src.models.dreamer import Dreamer
from src.trainer import Trainer

# from src.models.components.agent import Agent # AgentのPRがマージされたら追加
from tests.models.abc.dummy_classes import DummyAgent as Agent

obs_shape = (8,)
state_shape = (4,)
hidden_shape = (4,)
rnn_input_shape = (8,)

action_shape = (6,)

obs_enc = PE(in_channels=obs_shape[0], out_channels=state_shape[0])

obs_dec = CD(idim=state_shape[0], odim=obs_shape[0])
prior = Prior(hidden_shape[0], state_shape[0])
trans = Transition(hidden_shape[0], state_shape[0], action_shape[0], rnn_input_shape[0])

in_conv = PE(in_channels=obs_shape[0], out_channels=state_shape[0])
in_enc = PE(in_channels=state_shape[0], out_channels=state_shape[0])
enc_prj = PE(in_channels=state_shape[0], out_channels=state_shape[0] * 2)
enc_mods = (in_conv, in_enc, enc_prj)
ctrl = Ctrl(hidden_shape[0], state_shape[0], enc_mods, 5, hidden_shape[0], action_shape[0])

world = partial(World)

agent = partial(Agent)

w_opt = partial(SGD)
c_opt = partial(SGD)

dreamer_args = (trans, prior, obs_enc, obs_dec, ctrl, world, agent, w_opt, c_opt)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test__init__(device):
    # on cpu
    mod = Trainer(device=device)
    assert mod.num_episode is not None
    assert mod.collect_experience_interval is not None
    assert mod.batch_size is not None
    assert mod.chunk_size is not None
    assert mod.gradient_clip_value is not None
    assert mod.evaluation_interval is not None
    assert mod.model_save_interval is not None
    assert mod.device is not None
    assert mod.dtype is not None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_setup_model_attribute(device):
    trainer = Trainer(device=device)
    dreamer = Dreamer(*dreamer_args)
    trainer.setup_model_attribute(dreamer)

    assert dreamer.device == torch.device(trainer.device)
    assert dreamer.dtype == trainer.dtype
    assert dreamer.current_episode == 0
    assert dreamer.current_step == 0


# @pytest.mark.parametrize("device", ["cpu", "device"])
def test_fit():
    pass


def test_save_checkpoint():
    pass


def test_load_checkpoint():
    pass
