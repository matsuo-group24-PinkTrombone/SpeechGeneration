import torch

from tests.models.abc import dummy_classes as mod

batch_size = 8
hidden_shape = (16,)
state_shape = (32,)
action_shape = (7,)
voc_state_shape = (44,)
generated_sound_shape = (80, 5)
target_sound_shape = generated_sound_shape
embedded_obs_shape = (64,)
controller_hidden_shape = (16,)


hidden = torch.randn(batch_size, *hidden_shape)
state = torch.randn(batch_size, *state_shape)
action = torch.randn(batch_size, *action_shape)
voc_state = torch.rand(batch_size, *voc_state_shape)
generated_sound = torch.randn(batch_size, *generated_sound_shape) ** 2
target_sound = torch.randn(batch_size, *target_sound_shape) ** 2
embedded_obs = torch.randn(batch_size, *embedded_obs_shape)
controller_hidden = torch.randn(batch_size, *controller_hidden_shape)


def test_DummyTransition():
    cls = mod.DummyTransition

    obj = cls(hidden_shape)
    assert obj.forward(hidden, state, action).shape == hidden.shape
    assert obj.hidden_shape == hidden_shape


def test_DummyPrior():
    cls = mod.DummyPrior

    obj = cls(state_shape)
    assert obj.state_shape == state_shape
    out = obj.forward(hidden)
    assert isinstance(out, torch.distributions.Normal)
    assert out.sample().shape == state.shape


def test_DummyObservationEncoder():
    cls = mod.DummyObservationEncoder

    obj = cls(state_shape, embedded_obs_shape)

    assert obj.state_shape == state_shape
    assert obj.embed_observation((voc_state, generated_sound)).shape == embedded_obs.shape
    out = obj.encode(hidden, embedded_obs)
    assert isinstance(out, torch.distributions.Normal)
    assert out.sample().shape == state.shape


def test_DummyObservationDecoder():
    cls = mod.DummyObservationDecoder

    obj = cls(voc_state_shape, generated_sound_shape)

    out_v, out_g = obj.forward(hidden, state)
    assert out_v.shape == voc_state.shape
    assert out_g.shape == generated_sound.shape


def test_DummyController():
    cls = mod.DummyController
    obj = cls(action_shape, controller_hidden_shape)

    assert obj.controller_hidden_shape == controller_hidden_shape
    out_action, next_controller_hidden = obj.forward(
        hidden, state, target_sound, controller_hidden, True
    )
    assert out_action.shape == action.shape
    assert next_controller_hidden.shape == controller_hidden.shape
