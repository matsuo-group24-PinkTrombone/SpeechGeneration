import numpy as np
import pytest
from gym.spaces import Box
from pynktrombonegym.env import PynkTrombone

from src.env import normalize_action_range as mod
from tests.helpers.paths import sample_target_sound_file_pathes

cls = mod.NormalizeActionRange


def test__init__():
    base_env = PynkTrombone(sample_target_sound_file_pathes)
    low, high = -10.0, 10.0
    env = cls(base_env, low, high)

    assert env.low == low
    assert env.high == high

    for key in env.action_space.keys():
        box: Box = env.action_space[key]
        orig_box: Box = base_env.action_space[key]
        assert box.low == low
        assert box.high == high
        assert box.shape == orig_box.shape
        assert box.dtype == orig_box.dtype

    del base_env


@pytest.mark.parametrize("action_value", [-1.0, 0.0, 1.0])
def test_inv_normalized_action_range(action_value: float):
    base_env = PynkTrombone(sample_target_sound_file_pathes)
    low, high = -1.0, 1.0
    env = cls(base_env, low, high)

    dict_action = env.action_space.sample()
    for v in dict_action.values():
        v[:] = action_value

    inv_dict_action = env.inv_normalized_action_range(dict_action)
    for key in inv_dict_action.keys():
        inv_action = inv_dict_action[key]
        orig_box: Box = base_env.action_space[key]

        if action_value == -1.0:
            actual = orig_box.low
        elif action_value == 0.0:
            actual = (orig_box.low + orig_box.high) / 2
        elif action_value == 1.0:
            actual = orig_box.high
        else:
            raise ValueError(f"Unexpected value: {action_value}")

        np.testing.assert_allclose(inv_action, actual)


def test_step_action():
    base_env = PynkTrombone(sample_target_sound_file_pathes)
    low, high = -1.0, 1.0
    env = cls(base_env, low, high)

    sample_action = env.action_space.sample()
    env.step(sample_action)
