import os
from functools import partial
from tempfile import NamedTemporaryFile

import pytest
import torch
from pynktrombonegym.spaces import ObservationSpaceNames as OSN
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from src.env.array_action import ARRAY_ORDER as AO_act
from src.env.array_voc_state import VSON
from src.env.make_env import make_env
from src.models.dreamer import Dreamer
from src.trainer import Trainer

# from src.models.components.agent import Agent # AgentのPRがマージされたら追加
from tests.models.abc.dummy_classes import DummyAgent as Agent
from tests.models.abc.dummy_classes import DummyAgent as DA
from tests.models.abc.dummy_classes import DummyController as DC
from tests.models.abc.dummy_classes import DummyObservationDecoder as DOD
from tests.models.abc.dummy_classes import DummyObservationEncoder as DOE
from tests.models.abc.dummy_classes import DummyPrior as DP
from tests.models.abc.dummy_classes import DummyTransition as DT
from tests.models.abc.dummy_classes import DummyWorld as DW

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


def make_ckpt_path_and_tensorboard():
    p_dir = NamedTemporaryFile().name
    if os.path.exists(p_dir):
        os.makedirs(
            p_dir,
        )
    ckpt_path = p_dir
    tensorboard = SummaryWriter(p_dir)
    return ckpt_path, tensorboard


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test__init__(device):
    ckpt_destination, tb = make_ckpt_path_and_tensorboard()
    mod = Trainer(ckpt_destination, tb, device=device)
    assert mod.num_episode == 1
    assert mod.collect_experience_interval == 100
    assert mod.batch_size == 8
    assert mod.chunk_size == 64
    assert mod.gradient_clip_value == 100.0
    assert mod.evaluation_interval == 10
    assert mod.model_save_interval == 20
    assert mod.device == device
    assert mod.dtype == torch.float32


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_setup_model_attribute(device):
    ckpt_destination, tb = make_ckpt_path_and_tensorboard()
    trainer = Trainer(ckpt_destination, tb, device=device)
    dreamer = Dreamer(*dreamer_args)
    trainer.setup_model_attribute(dreamer)

    assert dreamer.device == torch.device(trainer.device)
    assert dreamer.dtype == trainer.dtype
    assert dreamer.current_episode == 0
    assert dreamer.current_step == 0
    assert dreamer.tensorboard == trainer.tensorboard


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fit(device):
    ckpt_destination, tb = make_ckpt_path_and_tensorboard()
    env = make_env(["data/sample_target_sounds"])
    trainer = Trainer(
        ckpt_destination, tb, device=device, collect_experience_interval=2, chunk_size=2
    )
    dreamer = Dreamer(*dreamer_args, imagination_horizon=1)
    rb = dreamer.configure_replay_buffer(env, buffer_size=4)
    trainer.setup_model_attribute(dreamer)
    trainer.fit(env, rb, dreamer)
    del env


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_save_checkpoint(device):
    ckpt_destination, tb = make_ckpt_path_and_tensorboard()
    dreamer = Dreamer(*dreamer_args)
    trainer = Trainer(ckpt_destination, tb, device=device, collect_experience_interval=2)
    wopt, copt = dreamer.configure_optimizers()
    save_path = os.path.join(ckpt_destination, "test.ckpt")
    trainer.save_checkpoint(save_path, dreamer, wopt, copt)
    assert os.path.exists(save_path)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_load_checkpoint(device):
    ckpt_destination, tb = make_ckpt_path_and_tensorboard()
    trainer = Trainer(ckpt_destination, tb, device=device, collect_experience_interval=2)
    dreamer = Dreamer(*dreamer_args)
    wopt, copt = dreamer.configure_optimizers()
    save_path = os.path.join(ckpt_destination, "test_load_checkpoint.ckpt")
    trainer.save_checkpoint(save_path, dreamer, wopt, copt)
    trainer.load_checkpoint(save_path, dreamer, wopt, copt)
