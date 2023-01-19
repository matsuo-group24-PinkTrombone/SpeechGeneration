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
        self.memory = self.init_memory()

    def push(self, examples: Dict[str, np.ndarray]):
        """You can push your data to ReplayBuffer with this method

        Args:
            examples (Dict[str, np.ndarray]): data to push 

        Raises:
            RuntimeError: This occurs when you input invalid space name which is not entered to __init__
        """
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
        """This method takes mini-batch from ReplayBuffer.

        Args:
            batch_size (int): The size of mini-batch.
            chunk_length (int): The time length of taken data.
            chunk_first (bool, optional): If True, returned sample's shape is (chunk_length, batch_size, *). Defaults to True.

        Returns:
            Dict[str, np.ndarray]: Mini-batch from ReplayBuffer.
        """
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
        """This method takes one sample from ReplayBuffer.

        Args:
            chunk_length (int): The time length of taken data.

        Returns:
            Dict[str, np.ndarray]: One sample from ReplayBuffer.
        """
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

    def init_memory(self):
        initialized_memory = {}
        for space_name, box in self.spaces.items():
            shape = (self.buffer_size,) + box.shape
            initialized_memory[space_name] = np.empty(shape)
        return initialized_memory
