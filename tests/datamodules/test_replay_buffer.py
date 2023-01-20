import numpy as np
import pytest
from gym.spaces import Box

from src.utils.replay_buffer import ReplayBuffer

spaces = {
    "space1": Box(-1, 1, (3,)),
    "space2": Box(-np.inf, 1, (3,)),
    "space3": Box(-1, np.inf, (3,)),
    "space4": Box(-np.inf, np.inf, (3,)),
    "space5": Box(0, 1, (1,), dtype=bool),
}
valid_buffer_size = 30000
invalid_buffer_size = -1


def test__init__():
    with pytest.raises(ValueError) as e:
        rb = ReplayBuffer(spaces, invalid_buffer_size)


def test_push():
    rb = ReplayBuffer(spaces, valid_buffer_size)
    num_adds = 64
    random_inputs = []
    invalid_input = {"invalid_space": Box(-np.inf, np.inf).sample()}

    with pytest.raises(RuntimeError) as e:
        rb.push(invalid_input)

    for i in range(num_adds):
        random_input = {space_name: box.sample() for space_name, box in spaces.items()}
        random_inputs.append(random_input)
        rb.push(random_input)

    for space_name in random_input.keys():
        for i in range(rb.current_index):
            assert (rb.memory[space_name][i] == random_inputs[i][space_name]).all()


def test_sample_chunk():
    pass


def test_sample():
    batch_size = 2
    chunk_length = 4
    num_adds = 64

    rb = ReplayBuffer(spaces, valid_buffer_size)
    for _ in range(num_adds):
        random_input = {space_name: box.sample() for space_name, box in spaces.items()}
        rb.push(random_input)

    batch_first_sample = rb.sample(batch_size, chunk_length, chunk_first=False)
    for space_name, box in spaces.items():
        assert batch_first_sample[space_name].shape == (batch_size, chunk_length, *box.shape)

    chunk_first_sample = rb.sample(batch_size, chunk_length, chunk_first=True)
    for space_name, box in spaces.items():
        assert chunk_first_sample[space_name].shape == (chunk_length, batch_size, *box.shape)


def test__len__():
    buffer_size = 10
    rb = ReplayBuffer(spaces, buffer_size)
    for _ in range(buffer_size - 1):
        random_input = {space_name: box.sample() for space_name, box in spaces.items()}
        rb.push(random_input)
    # Before rb.current_index reached to buffer_size,
    # len(rb) should be equeal to self.current_index
    assert len(rb) == buffer_size - 1

    # Once rb.current_index reached to buffer_size,
    # len(rb) always returns buffer_size
    rb.push(random_input)
    assert len(rb) == buffer_size
    rb.push(random_input)
    assert len(rb) == buffer_size


def test_init_values():
    rb = ReplayBuffer(spaces, valid_buffer_size)
    for space_name, box in spaces.items():
        assert rb.memory.get(space_name) is not None
        assert len(rb.memory.get(space_name)[-1]) == box.shape[0]
