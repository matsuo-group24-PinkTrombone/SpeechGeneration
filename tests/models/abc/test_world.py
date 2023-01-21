import torch

from src.models.abc import world as mod
from tests.models.abc.dummy_classes import (
    DummyObservationDecoder,
    DummyObservationEncoder,
    DummyPrior,
    DummyTransition,
)

batch_size = 8
hidden_shape = (16,)
state_shape = (32,)
action_shape = (7,)
voc_state_shape = (44,)
generated_sound_shape = (80, 5)
target_sound_shape = generated_sound_shape
embedded_obs_shape = (64,)


hidden = torch.randn(batch_size, *hidden_shape)
state = torch.randn(batch_size, *state_shape)
action = torch.randn(batch_size, *action_shape)
voc_state = torch.rand(batch_size, *voc_state_shape)
generated_sound = torch.randn(batch_size, *generated_sound_shape) ** 2
target_sound = torch.randn(batch_size, *target_sound_shape) ** 2
embedded_obs = torch.randn(batch_size, *embedded_obs_shape)

dummy_trans = DummyTransition(hidden_shape)
dummy_prior = DummyPrior(state_shape)
dummy_obs_enc = DummyObservationEncoder(state_shape, embedded_obs_shape)
dummy_obs_dec = DummyObservationDecoder(voc_state_shape, generated_sound_shape)


def test_World():
    cls = mod.World

    obj = cls(dummy_trans, dummy_prior, dummy_obs_enc, dummy_obs_dec)

    next_state_prior, next_state_post, next_hidden = obj.forward(
        hidden, state, action, next_obs=(voc_state, generated_sound)
    )

    assert isinstance(next_state_prior, torch.distributions.Normal)
    assert isinstance(next_state_post, torch.distributions.Normal)
    assert next_state_prior.sample().shape == next_state_post.sample().shape
    assert next_state_prior.sample().shape == state.shape
    assert next_hidden.shape == hidden.shape
