from typing import Dict

import numpy as np
from gym.spaces import Box


class ReplayBuffer:
    def __init__(self, spaces: Dict[str, Box], buffer_size: int):
        if buffer_size < 0:
            raise ValueError(f"Input is {buffer_size} but buffer size must be greater than 0")

        self.spaces = spaces
        self.buffer_size = buffer_size
        self.current_index = 0
        self.is_capacity_reached = False
        self.memory = self.init_values()

    def push(self, examples: Dict[str, np.ndarray]):
        invalid_space_names = set(examples.keys()).difference(self.spaces.keys())
        if not invalid_space_names == set():
            raise RuntimeError(f"space names {invalid_space_names} are invalid space names")

        for space_name, value in examples.items():
            self.memory[space_name][self.current_index] = value

        if not self.is_capacity_reached:
            self.is_capacity_reached = (self.current_index + 1) >= (self.buffer_size)
        self.current_index = (self.current_index + 1) % self.buffer_size

    def sample(
        self, batch_size: int, chunk_length: int, chunk_first: bool = True
    ) -> Dict[str, np.ndarray]:
        minibatch = {space_name: [] for space_name in self.spaces.keys()}
        for _ in range(batch_size):
            data = self.sample_chunk(chunk_length)
            for space_name, value in data.items():
                minibatch[space_name].append(value)

        for space_name, value in minibatch.items():
            data_np = np.stack(minibatch[space_name])
            if chunk_first:
                data_np = data_np.transpose(1, 0, 2)
            minibatch[space_name] = data_np
        return minibatch

    def sample_chunk(
        self,
        chunk_length: int,
    ) -> Dict[str, np.ndarray]:
        # TODO:self.current_indexを考慮した実装
        if self.is_capacity_reached:
            max_index = self.buffer_size
        else:
            max_index = self.current_index
        start_index = np.random.randint(0, max_index)
        sampled_indice = np.arange(chunk_length) + start_index
        sampled_data = {key: value[sampled_indice] for key, value in self.memory.items()}
        return sampled_data

    def __len__(self):
        if self.is_capacity_reached:
            return self.buffer_size
        else:
            return self.current_index

    def init_values(self):
        initialized_memory = {}
        for space_name, box in self.spaces.items():
            shape = (self.buffer_size,) + box.shape
            initialized_memory[space_name] = np.empty(shape)
        return initialized_memory
