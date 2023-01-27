import glob
import pathlib
from functools import partial

import numpy as np
import pytest
import torch
from gym.spaces import Box
from pynktrombonegym.env import PynkTrombone as PT
from pynktrombonegym.spaces import ObservationSpaceNames as OSN
from torch.optim import AdamW

from src.datamodules import buffer_names
from src.datamodules.replay_buffer import ReplayBuffer
from src.env.array_action import ARRAY_ORDER as AO_act
from src.env.array_action import ArrayAction as AA
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
env = AA(AVS(PT(target_files)))

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

world_opt = AdamW
ctrl_opt = AdamW


bf_size = 32
bf_space = {
    buffer_names.ACTION: Box(-np.inf, np.inf, action_shape),
    buffer_names.DONE: Box(-np.inf, np.inf, (1,)),
    buffer_names.GENERATED_SOUND: Box(-np.inf, np.inf, gen_sound_shape),
    buffer_names.TARGET_SOUND: Box(-np.inf, np.inf, tgt_sound_shape),
    buffer_names.VOC_STATE: Box(-np.inf, np.inf, voc_stats_shape),
}
args = (trans, prior, obs_enc, obs_dec, ctrl, DW, DA, world_opt, ctrl_opt)


def test__init__():
    model = Dreamer(*args)


def test_configure_optimizers():
    model = Dreamer(*args)
    opt1, opt2 = model.configure_optimizers()


@pytest.mark.parametrize("num_steps", [1, 2, 3])
def test_collect_experiences(num_steps):
    rb = ReplayBuffer(bf_space, bf_size)
    model = Dreamer(*args, num_collect_experience_steps=num_steps)
    model.collect_experiences(env, rb)
    assert rb.current_index == num_steps


def test_world_training_step():
    model = Dreamer(*args, num_collect_experience_steps=128)
    rb = ReplayBuffer(bf_space, bf_size)
    _, __ = model.configure_optimizers()
    model.collect_experiences(env, rb)
    experience = rb.sample(1, chunk_length=16)
    print(rb.is_capacity_reached)
    loss_dict, experience = model.world_training_step(experience)
    assert experience.get("hiddens") is not None
    assert experience.get("states") is not None


def test_controller_training_step():
    pass


def test_evaluation_step():
    pass
