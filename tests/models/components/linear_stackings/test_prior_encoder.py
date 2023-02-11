import pytest
import torch

from src.models.components.linear_stackings.prior_encoder import Prior

hidden_dim = 10
invalid_hidden_dim = -1
state_dim = 5
invalid_state_dim = -1
batch_size = 4
stacked_linear_hidden_size = 16
num_layers = 2


def test__init__():
    Prior(hidden_dim, state_dim, stacked_linear_hidden_size, num_layers)

    with pytest.raises(AssertionError):
        model = Prior(invalid_hidden_dim, state_dim, stacked_linear_hidden_size)
    with pytest.raises(AssertionError):
        model = Prior(hidden_dim, invalid_state_dim, stacked_linear_hidden_size)


def test_forward():
    model = Prior(hidden_dim, state_dim, stacked_linear_hidden_size)
    # single input
    random_input = torch.rand((hidden_dim,), requires_grad=True)
    state_distribution = model(random_input)
    assert state_distribution.sample().shape == (state_dim,)
    assert state_distribution.rsample().requires_grad

    # batch input
    random_input = torch.rand((batch_size, hidden_dim), requires_grad=True)
    state_distribution = model(random_input)
    assert state_distribution.sample().shape == (batch_size, state_dim)
    assert state_distribution.rsample().requires_grad


def test_state_shape():
    model = Prior(hidden_dim, state_dim, stacked_linear_hidden_size)
    assert model.state_shape == (state_dim,)
