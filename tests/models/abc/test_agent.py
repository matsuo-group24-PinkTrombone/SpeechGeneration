import copy

import pytest
import torch
from torch import Tensor

from src.models.abc import agent as mod
from tests.models.abc.dummy_classes import (
    DummyController,
    DummyObservationEncoder,
    DummyTransition,
)

batch_size = 1  # constant 1.
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

dummy_trans = DummyTransition(hidden_shape)
dummy_obs_enc = DummyObservationEncoder(state_shape, embedded_obs_shape)
dummy_controller = DummyController(action_shape, controller_hidden_shape)


def test_Agent():
    cls = mod.Agent
    dum_con = copy.deepcopy(dummy_controller)  # ignore buffer modification
    dum_trn = copy.deepcopy(dummy_trans)  # ignore buffer modification

    with pytest.raises(TypeError) as e:
        cls(dum_con, dum_trn, dummy_obs_enc)

    class C(cls):
        def explore(self, obs: tuple[Tensor, Tensor], target: Tensor) -> Tensor:
            return self.act(obs, target, True)

    obj = C(dum_con, dum_trn, dummy_obs_enc)

    assert "_hidden_of_agent" in dum_trn._buffers
    assert "_controller_hidden_of_agent" in dum_con._buffers
    assert dum_trn._buffers["_hidden_of_agent"] is obj.hidden
    assert dum_con._buffers["_controller_hidden_of_agent"] is obj.controller_hidden

    assert obj.hidden.shape == (1, *hidden_shape)
    assert obj.controller_hidden.shape == (1, *controller_hidden_shape)
    assert (obj.hidden == 0.0).all()
    assert (obj.controller_hidden == 0.0).all()

    next_action = obj.act(
        obs=(voc_state, generated_sound), target=target_sound, probabilistic=False
    )

    assert not (obj.hidden == 0.0).all()
    assert not (obj.controller_hidden == 0.0).all()
    assert next_action.shape == action.shape
    obj.explore(obs=(voc_state, generated_sound), target=target_sound)

    obj.reset()
    assert (obj.hidden == 0.0).all()
    assert (obj.controller_hidden == 0.0).all()
