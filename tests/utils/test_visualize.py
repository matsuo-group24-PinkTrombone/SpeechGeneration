from src.utils.visualize import make_spectrogram_figure, visualize_model_approximation
import glob
import os
import pathlib
from datetime import datetime
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pytest
import torch
from gym.spaces import Box
from pynktrombonegym.spaces import ObservationSpaceNames as OSN

from src.env.make_env import make_env
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from src.datamodules import buffer_names
from src.datamodules.replay_buffer import ReplayBuffer
from src.env.array_action import ARRAY_ORDER as AO_act

from src.env.array_voc_state import ARRAY_ORDER as AO_voc
from src.env.array_voc_state import VSON
from src.utils.visualize import make_spectrogram_figure

from src.models.dreamer import Dreamer
from tests.models.abc.dummy_classes import DummyAgent as DA
from tests.models.abc.dummy_classes import DummyController as DC
from tests.models.abc.dummy_classes import DummyObservationDecoder as DOD
from tests.models.abc.dummy_classes import DummyObservationEncoder as DOE
from tests.models.abc.dummy_classes import DummyPrior as DP
from tests.models.abc.dummy_classes import DummyTransition as DT
from tests.models.abc.dummy_classes import DummyWorld as DW

target_file_path = str(pathlib.Path(__file__).parents[2].joinpath("data/sample_target_sounds/"))
env = make_env([target_file_path])

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

tb_log_dir = f"logs/test_dreamer/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def test_make_spectrogram_figure():
    model = Dreamer(*args)
    model.tensorboard = SummaryWriter(os.path.join(tb_log_dir, "test_log"))
    shape = (128, 256)
    gen_spect = np.random.rand(*shape)
    tgt_spect = np.random.rand(*shape)
    pred_gen_spect = np.random.rand(*shape)
    tag = "evaluation_step/mel_spect/test"
    for i in range(3):
        fig: plt.figure = make_spectrogram_figure(tgt_spect, gen_spect, pred_gen_spect)
        model.tensorboard.add_figure(tag, fig, global_step=i+1)
    

def test_visualize_model_approximation():
    env = make_env([target_file_path])
    model = Dreamer(*args)
    tb=SummaryWriter(os.path.join(tb_log_dir, "test_log"))
    world = model.world
    agent = model.agent

    for i in range(3):
        visualize_model_approximation(
            world,
            agent,
            env,
            tb,
            "evaluation_step/mel_spect/test",
            i,
        )
    del env