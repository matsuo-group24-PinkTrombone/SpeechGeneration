import glob

import numpy as np
from gym.spaces import Box
from pynktrombonegym.env import PynkTrombone

from src.env import array_action as mod

cls = mod.ArrayAction

sample_target_sound_file_paths = glob.glob("data/sample_target_sounds/*")


def test_array_order():
    order = mod.ARRAY_ORDER

    order_set = set(order)
    assert len(order_set) == len(order)

    for k, v in mod.ASN.__dict__.items():
        if not k.startswith("__"):
            assert v in order_set


def test__init__():
    base_env = PynkTrombone(sample_target_sound_file_paths)
    env = cls(base_env)

    assert isinstance(env.action_space, Box)
    assert env.action_space.shape == (len(mod.ARRAY_ORDER),)
    assert env.action_space.dtype == np.float32

    for i, name in enumerate(mod.ARRAY_ORDER):
        np.testing.assert_allclose(base_env.action_space[name].low, env.action_space.low[i])

        np.testing.assert_allclose(base_env.action_space[name].high, env.action_space.high[i])


def test_array_to_dict():
    base_env = PynkTrombone(sample_target_sound_file_paths)
    env = cls(base_env)

    array_action = env.action_space.sample()
    dict_action = env.array_to_dict(array_action)

    for i, name in enumerate(mod.ARRAY_ORDER):
        np.testing.assert_allclose(dict_action[name], array_action[i])
        assert dict_action[name].shape == (1,)


def test_step_action():

    base_env = PynkTrombone(sample_target_sound_file_paths)
    env = cls(base_env)
    env.step(env.action_space.sample())
