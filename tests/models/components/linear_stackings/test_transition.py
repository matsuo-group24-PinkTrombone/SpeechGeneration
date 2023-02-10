import torch

from src.models.components.linear_stackings.transition import Transition

hidden_size = 10
action_size = 8
state_size = 8
input_size = 8
batch_size = 4
stacked_linear_hidden_size = 16
num_layers = 3

args = {
    "hidden_size": hidden_size,
    "action_size": action_size,
    "state_size": state_size,
    "input_size": input_size,
    "stacked_linear_hidden_size": stacked_linear_hidden_size,
    "num_layers": num_layers,
}


def test__init__():
    model = Transition(**args)

    assert model.hidden_size == hidden_size
    assert model.action_size == action_size
    assert model.state_size == state_size
    assert model.input_size == input_size
    assert model.stacked_linear_hidden_size == stacked_linear_hidden_size


def test_forward():
    model = Transition(**args)

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


def test_hidden_shape():
    model = Transition(**args)
    assert model.hidden_shape == (hidden_size,)
