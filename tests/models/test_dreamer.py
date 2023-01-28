import glob
import pathlib
from functools import partial

import numpy as np
import pytest
import torch
from gym.spaces import Box
from pynktrombonegym.wrappers import Log1pMelSpectrogram as L1MS
from pynktrombonegym.wrappers import ActionByAcceleration as ABA
from src.env.normalize_action_range import NormalizeActionRange as NAR
from src.env.array_action import ArrayAction as AA
from src.env.array_voc_state import ArrayVocState as AVS
from pynktrombonegym.spaces import ObservationSpaceNames as OSN
from torch.optim import SGD

from src.datamodules import buffer_names
from src.datamodules.replay_buffer import ReplayBuffer
from src.env.array_action import ARRAY_ORDER as AO_act

from src.env.array_voc_state import ARRAY_ORDER as AO_voc
from src.env.array_voc_state import VSON
from src.env.array_voc_state import ArrayVocState as AVS
from src.models.dreamer import Dreamer
from tests.models.abc.dummy_classes import DummyAgent as DA
from tests.models.abc.dummy_classes import DummyController as DC
from tests.models.abc.dummy_classes import DummyObservationDecoder as DOD
from tests.models.abc.dummy_classes import DummyObservationEncoder as DOE
from tests.models.abc.dummy_classes import DummyPrior as DP
from tests.models.abc.dummy_classes import DummyTransition as DT
from tests.models.abc.dummy_classes import DummyWorld as DW

target_file_path = pathlib.Path(__file__).parents[2].joinpath("data/sample_target_sounds/*.wav")
target_files = glob.glob(str(target_file_path))
env = AVS(AA(NAR(ABA(L1MS(target_files), action_scaler=1.0))))

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
d_agent = partial(DA)

world_opt = partial(SGD, lr=1e-3)
ctrl_opt = partial(SGD, lr=1e-3)


bf_size = 32
bf_space = {
    buffer_names.ACTION: Box(-np.inf, np.inf, action_shape),
    buffer_names.DONE: Box(-np.inf, np.inf, (1,)),
    buffer_names.GENERATED_SOUND: Box(-np.inf, np.inf, gen_sound_shape),
    buffer_names.TARGET_SOUND: Box(-np.inf, np.inf, tgt_sound_shape),
    buffer_names.VOC_STATE: Box(-np.inf, np.inf, voc_stats_shape),
}
args = (trans, prior, obs_enc, obs_dec, ctrl, d_world, d_agent, world_opt, ctrl_opt)
del env

def world_training_step(model, env):
    rb = ReplayBuffer(bf_space, bf_size)
    _, __ = model.configure_optimizers()
    model.collect_experiences(env, rb)
    experience = rb.sample(1, chunk_length=16)
    loss_dict, experience = model.world_training_step(experience)
    return loss_dict, experience


def test__init__():
    model = Dreamer(*args)


def test_configure_optimizers():
    model = Dreamer(*args)
    opt1, opt2 = model.configure_optimizers()


@pytest.mark.parametrize("num_steps", [1, 2, 3])
def test_collect_experiences(num_steps):
    env = AVS(AA(NAR(ABA(L1MS(target_files), action_scaler=1.0))))
    rb = ReplayBuffer(bf_space, bf_size)
    model = Dreamer(*args, num_collect_experience_steps=num_steps)
    model.collect_experiences(env, rb)
    assert rb.current_index == num_steps
    del env


def test_world_training_step():
    env = AVS(AA(NAR(ABA(L1MS(target_files), action_scaler=1.0))))
    model = Dreamer(*args, num_collect_experience_steps=128)
    loss_dict, experience = world_training_step(model, env)
    assert experience.get("hiddens") is not None
    assert experience.get("states") is not None
    assert loss_dict.get("loss") is not None
    del env


def test_controller_training_step():
    # World Training Step
    env = AVS(AA(NAR(ABA(L1MS(target_files), action_scaler=1.0))))
    model = Dreamer(*args, imagination_horizon=4)
    _, experience = world_training_step(model, env)
    loss_dict, _ = model.controller_training_step(experience)
    assert loss_dict.get("loss") is not None
    del env


def test_evaluation_step():
    pass

