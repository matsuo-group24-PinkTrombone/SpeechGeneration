from functools import partial
from tempfile import NamedTemporaryFile
import os

import pytest
import torch
from torch.optim import SGD

from src.env.array_action import ARRAY_ORDER as AO_act
from src.env.make_env import make_env
from src.models.dreamer import Dreamer
from src.trainer import Trainer
from src.env.array_voc_state import VSON
from pynktrombonegym.spaces import ObservationSpaceNames as OSN
from tests.models.abc.dummy_classes import DummyAgent as DA
from tests.models.abc.dummy_classes import DummyController as DC
from tests.models.abc.dummy_classes import DummyObservationDecoder as DOD
from tests.models.abc.dummy_classes import DummyObservationEncoder as DOE
from tests.models.abc.dummy_classes import DummyPrior as DP
from tests.models.abc.dummy_classes import DummyTransition as DT
from tests.models.abc.dummy_classes import DummyWorld as DW

# from src.models.components.agent import Agent # AgentのPRがマージされたら追加
from tests.models.abc.dummy_classes import DummyAgent as Agent

env = make_env(["data/sample_target_sounds"])

hidden_shape = (16,)
ctrl_hidden_shape = (16,)
state_shape = (8,)
action_shape = (len(AO_act),)

obs_space = env.observation_space
voc_stats_shape = obs_space[VSON.VOC_STATE].shape
rnn_input_shape = (8,)

gen_sound_shape = obs_space[OSN.GENERATED_SOUND_SPECTROGRAM].shape
tgt_sound_shape = obs_space[OSN.TARGET_SOUND_SPECTROGRAM].shape
obs_shape = (24,)

obs_enc = DOE(state_shape, obs_shape)
obs_dec = DOD(
    voc_stats_shape,
    gen_sound_shape,
)
prior = DP(state_shape)
trans = DT(hidden_shape)
ctrl = DC(action_shape, ctrl_hidden_shape)


d_world = partial(DW)
d_agent = partial(DA, action_shape=action_shape)

world_opt = partial(SGD, lr=1e-3)
ctrl_opt = partial(SGD, lr=1e-3)

dreamer_args = (trans, prior, obs_enc, obs_dec, ctrl, d_world, d_agent, world_opt, ctrl_opt)
del env


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test__init__(device):
    log_dir = os.path.join(NamedTemporaryFile().name, "tensorboard")
    mod = Trainer(log_dir=log_dir, device=device)
    assert mod.num_episode is not None
    assert mod.collect_experience_interval is not None
    assert mod.batch_size is not None
    assert mod.chunk_size is not None
    assert mod.gradient_clip_value is not None
    assert mod.evaluation_interval is not None
    assert mod.model_save_interval is not None
    assert mod.device is not None
    assert mod.dtype is not None
    del log_dir


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_setup_model_attribute(device):
    log_dir = os.path.join(NamedTemporaryFile().name, "tensorboard")
    trainer = Trainer(log_dir=log_dir, device=device)
    dreamer = Dreamer(*dreamer_args)
    trainer.setup_model_attribute(dreamer)

    assert dreamer.device == torch.device(trainer.device)
    assert dreamer.dtype == trainer.dtype
    assert dreamer.current_episode == 0
    assert dreamer.current_step == 0
    del log_dir


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fit(device):
    log_dir = os.path.join(NamedTemporaryFile().name, "tensorboard")
    env = make_env(["data/sample_target_sounds"])
    trainer = Trainer(log_dir=log_dir, device=device, collect_experience_interval=2, chunk_size=2)
    dreamer = Dreamer(*dreamer_args)
    rb = dreamer.configure_replay_buffer(env, buffer_size=4)
    trainer.setup_model_attribute(dreamer)
    trainer.fit(env, rb, dreamer)
    del env, log_dir


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_save_checkpoint(device):
    log_dir = os.path.join(NamedTemporaryFile().name, "tensorboard")
    dreamer = Dreamer(*dreamer_args)
    env = make_env(["data/sample_target_sounds"])
    trainer = Trainer(log_dir=log_dir, device=device, collect_experience_interval=2)
    wopt, copt = dreamer.configure_optimizers()
    ckpt = NamedTemporaryFile()
    trainer.save_checkpoint(ckpt.name, dreamer, wopt, copt)
    del log_dir

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_load_checkpoint(device):
    log_dir = os.path.join(NamedTemporaryFile().name, "tensorboard")
    trainer = Trainer(log_dir=log_dir, device=device, collect_experience_interval=2)
    dreamer = Dreamer(*dreamer_args)
    wopt, copt = dreamer.configure_optimizers()
    ckpt = NamedTemporaryFile()
    trainer.save_checkpoint(ckpt.name, dreamer, wopt, copt)
    trainer.load_checkpoint(ckpt.name, dreamer, wopt, copt)
    del log_dir
