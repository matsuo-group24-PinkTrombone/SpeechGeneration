import copy
from collections import OrderedDict

import gym
import numpy as np
from gym import spaces
from pynktrombonegym.spaces import ObservationSpaceNames as OSN


class VocStateObsNames(OSN):
    VOC_STATE = "voc_state"


ARRAY_ORDER = [
    OSN.FREQUENCY,
    OSN.PITCH_SHIFT,
    OSN.TENSENESS,
    OSN.CURRENT_TRACT_DIAMETERS,
    OSN.NOSE_DIAMETERS,
]

VSON = VocStateObsNames


class ArrayVocState(gym.ObservationWrapper):
    """Observation Wrapper for making voc_state as array."""

    def __init__(self, env: gym.Env, new_step_api: bool = False):
        """
        Args:
            env (gym.Env): PynkTromboneGym Env or its wrapper.
            voc (Voc): Vocal Tract Model class.
            new_step_api (bool): gym api.
        """
        super().__init__(env, new_step_api)

        self.observation_space = self.define_observation_space()

    observation_space: spaces.Dict

    def define_observation_space(self) -> spaces.Dict:
        """Re-define observation space of environment."""
        orig_space: spaces.Dict = copy.deepcopy(self.env.observation_space)

        lows, highs = [], []
        for name in ARRAY_ORDER:
            orig_box: spaces.Box = orig_space[name]
            lows.append(np.full(orig_box.shape, orig_box.low, dtype=orig_box.dtype))
            highs.append(np.full(orig_box.shape, orig_box.high, dtype=orig_box.dtype))

        lows = np.concatenate(lows)
        highs = np.concatenate(highs)

        box = spaces.Box(lows, highs, dtype=np.float32)

        orig_space[VSON.VOC_STATE] = box
        return orig_space

    def make_voc_state_array(self, original_obs: OrderedDict) -> np.ndarray:
        """Making `voc_state` array from original observation.
        Args:
            original_obs (OrderedDict): Original observation of environment.

        Returns:
            voc_state (np.ndarray): voc_state 1d array.
        """
        voc_state = []
        for name in ARRAY_ORDER:
            voc_state.append(original_obs[name])
        voc_state = np.concatenate(voc_state)

        return voc_state

    def observation(self, observation: OrderedDict) -> OrderedDict:
        """Make and add `voc_state` array to original observation.
        Args:
            observation (OrderedDict): Original Observation

        Returns:
            new_observation (OrderedDict): Added `voc_state`.
        """
        observation[VSON.VOC_STATE] = self.make_voc_state_array(observation)

        return observation
