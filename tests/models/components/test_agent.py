import pytest
import torch

from src.models.components import agent as mod
from tests.models.abc.dummy_classes import (
    DummyController,
    DummyObservationEncoder,
    DummyTransition,
)

cls = mod.Agent

batch_size = 1
action_shape = (7,)
controller_hidden_shape = (16,)
hidden_shape = (32,)
state_shape = (8,)
embedded_obs_shape = (64,)
controller = DummyController(action_shape, controller_hidden_shape)
transition = DummyTransition(hidden_shape)
obs_encoder = DummyObservationEncoder(state_shape, embedded_obs_shape)

generated_spect = torch.randn(batch_size, 80, 5).abs()
voc_state = torch.randn(batch_size, 75)
obs = (voc_state, generated_spect)
target = torch.randn_like(generated_spect).abs()


@pytest.mark.parametrize("action_noise_ratio", [0.0, 0.5, 1.0])
def test_explorer(action_noise_ratio):

    agent = cls(controller, transition, obs_encoder, action_noise_ratio)

    # Sanity checks
    out = agent.explore(obs, target)
    assert out.shape == (batch_size, *action_shape)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)
