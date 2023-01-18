import pytest
import numpy as np

from src.utils.replay_buffer import ReplayBuffer
from gym.spaces import Box

spaces = {
    "space1": Box(-1, 1, (3,)),
    "space2": Box(-np.inf, 1, (3,)),
    "space3": Box(-1, np.inf, (3,)),
    "space4": Box(-np.inf, np.inf, (3,)),
}
valid_buffer_size = 30000
invalid_buffer_size = -1

def test__init__():
    with pytest.raises(ValueError) as e:
        rb = ReplayBuffer(spaces, invalid_buffer_size)

    
def test_push():
    pass

def test_sample_chunk():
    pass

def test_sample():
    pass

def test__len__():
    pass

def test_init_values():
    rb = ReplayBuffer(spaces, valid_buffer_size)
    for space_name, box in spaces.items():
        assert rb.memory.get(space_name) is not None
        assert len(rb.memory.get(space_name)) == box.shape[0]