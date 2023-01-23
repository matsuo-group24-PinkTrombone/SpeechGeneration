from collections import OrderedDict

import gym
import numpy as np
from gym import spaces


class NormalizeActionRange(gym.ActionWrapper):
    """Normalize Action Range of Action Wrapper.

    Fix action range to [-1, 1].
    """

    def __init__(self, env: gym.Env, low: float = -1.0, high: float = 1.0) -> None:
        """
        Args:
            env (gym.Env): PynkTromboneGym Env or its wrapper.
            low (float): The lowest value of normalized action.
            high (float): The highest value of normalized action.
        """
        super().__init__(env)
        self.low = low
        self.high = high

        self.action_space = self.define_action_space()

    action_space: spaces.Dict

    def define_action_space(self):
        """Re define action space of environment."""
        orig_space: spaces.Dict = self.env.action_space

        new_space = dict()
        for (name, box) in orig_space.items():
            new_space[name] = spaces.Box(self.low, self.high, box.shape, dtype=box.dtype)

        return spaces.Dict(new_space)

    def inv_normalized_action_range(self, dict_action: OrderedDict) -> OrderedDict:
        """Inverse normalized action range for original environment.
        Args:
            dict_action (spaces.Dict): Normalized dict action.

        Returns:
            inv_dict_action (spaces.Dict): Inv-normalized dict action.
        """
        inv_dict_action = OrderedDict()

        for (name, action) in dict_action.items():
            action: np.ndarray
            sp: spaces.Box = self.env.action_space[name]
            inv = (action * (sp.high - sp.low) + (sp.high + sp.low)) / 2
            inv_dict_action[name] = inv

        return inv_dict_action

    def action(self, action: OrderedDict) -> OrderedDict:
        """Wraps action."""
        return self.inv_normalized_action_range(action)
