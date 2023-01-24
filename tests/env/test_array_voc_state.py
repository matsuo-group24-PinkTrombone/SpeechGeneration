import glob

import numpy as np
from gym.spaces import Box
from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.spaces import ObservationSpaceNames as OSN

from src.env import array_voc_state as mod

cls = mod.ArrayVocState
name_cls = mod.VocStateObsNames
array_order = mod.ARRAY_ORDER
sample_target_sound_file_paths = glob.glob("data/sample_target_sounds/*")


def test_name_cls():
    assert name_cls.VOC_STATE == "voc_state"


def test_array_order():
    order_set = set(array_order)
    assert len(order_set) == len(array_order)

    assert array_order[0] == OSN.FREQUENCY
    assert array_order[1] == OSN.PITCH_SHIFT
    assert array_order[2] == OSN.TENSENESS
    assert array_order[3] == OSN.CURRENT_TRACT_DIAMETERS
    assert array_order[4] == OSN.NOSE_DIAMETERS

    assert len(array_order) == 5


def test__init__():
    base_env = PynkTrombone(sample_target_sound_file_paths)
    env = cls(base_env)

    start = 0
    end = start

    voc_state_box: Box = env.observation_space[name_cls.VOC_STATE]

    for name in array_order:
        orig_box: Box = base_env.observation_space[name]
        length = orig_box.shape[0]
        end += length
        np.testing.assert_allclose(voc_state_box.low[start:end], orig_box.low)
        np.testing.assert_allclose(voc_state_box.high[start:end], orig_box.high)

        start += length

    assert voc_state_box.shape == (end,)
    assert voc_state_box.dtype == np.float32


def test_make_voc_state_array():
    base_env = PynkTrombone(sample_target_sound_file_paths)
    env = cls(base_env)

    obs = base_env.observation_space.sample()

    out = env.make_voc_state_array(obs)

    assert isinstance(out, np.ndarray)
    assert out.shape == env.observation_space[name_cls.VOC_STATE].shape


def test_observation():
    base_env = PynkTrombone(sample_target_sound_file_paths)
    env = cls(base_env)

    obs = base_env.observation_space.sample()
    out = env.observation(obs)

    np.testing.assert_allclose(out[name_cls.VOC_STATE], env.make_voc_state_array(obs))


def test_reset_and_step():
    base_env = PynkTrombone(sample_target_sound_file_paths)
    env = cls(base_env)

    obs = env.reset()
    assert name_cls.VOC_STATE in obs

    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
    assert name_cls.VOC_STATE in obs
