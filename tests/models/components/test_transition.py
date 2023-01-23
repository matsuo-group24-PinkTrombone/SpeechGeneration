import torch

from src.models.components.transition import Transition

hidden_size = 10
action_size = 8
state_size = 8
input_size = 8
batch_size = 4


def test_forward():
    model = Transition(hidden_size, state_size, action_size, input_size)

    # single input
    random_state = torch.rand(state_size)
    random_action = torch.rand(action_size)
    random_hidden = torch.rand(hidden_size)
    next_hidden = model(random_hidden, random_state, random_action)
    assert next_hidden.shape == (hidden_size,)

    # batch input
    random_state = torch.rand((batch_size, state_size))
    random_action = torch.rand((batch_size, action_size))
    random_hidden = torch.rand((batch_size, hidden_size))
    next_hidden = model(random_hidden, random_state, random_action)
    assert next_hidden.size(-1) == hidden_size
    assert next_hidden.size(0) == batch_size
