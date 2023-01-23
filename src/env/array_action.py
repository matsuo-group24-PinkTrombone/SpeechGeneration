from collections import OrderedDict

import gym
import numpy as np
from gym import spaces
from pynktrombonegym.spaces import ActionSpaceNames as ASN

ARRAY_ORDER = [
    ASN.PITCH_SHIFT,
    ASN.TENSENESS,
    ASN.TRACHEA,
    ASN.EPIGLOTTIS,
    ASN.VELUM,
    ASN.TONGUE_INDEX,
    ASN.TONGUE_DIAMETER,
    ASN.LIPS,
]


class ArrayAction(gym.ActionWrapper):
    """Convert dict action space to array (Box) space."""

    def __init__(self, env: gym.Env, new_step_api: bool = False):
        """
        Args:
            env (gym.Env): PynkTromboneGym Env or its wrapper.
            new_step_api (bool): Gym wrapper argument.
        """
        super().__init__(env, new_step_api)

        self.action_space = self.define_action_space()

    action_space: spaces.Box

    def define_action_space(self) -> spaces.Box:
        """Re-define action space of environment."""
        orig_space: spaces.Dict = self.env.action_space

        lows, highs = [], []
        for name in ARRAY_ORDER:
            orig_box: spaces.Box = orig_space[name]
            lows.append(orig_box.low)
            highs.append(orig_box.high)

        lows = np.concatenate(lows)
        highs = np.concatenate(highs)

        box = spaces.Box(lows, highs, dtype=np.float32)

        return box

    def array_to_dict(self, array_action: np.ndarray) -> OrderedDict:
        """Convert array action to dict action.
        Args:
            array_action (np.ndarray): Array of action parameters.

        Returns:
            dict_action (OrderedDict): Dictionary action for original environment.
        """

        assert len(array_action) == len(ARRAY_ORDER)
        array_action = array_action.reshape(-1, 1)
        dict_action = OrderedDict()
        for i, name in enumerate(ARRAY_ORDER):
            dict_action[name] = array_action[i]

        return dict_action

    def action(self, action: np.ndarray) -> OrderedDict:
        return self.array_to_dict(action)
