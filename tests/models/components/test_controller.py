import torch
import pytest

from src.models.components.controller import Controller

@pytest.mark.parametrize(
    """
    batch_size,
    hidden_dim,
    state_dim,
    target_dim,
    c_hidden_dim,
    action_dim,
    prob
    """,
    [
        (1,192,192,192,192,7,False),
        (5,12,12,12,12,44,True),

    ]
)
def test_controller(
    batch_size:int,
    hidden_dim:int,
    state_dim:int,
    target_dim:int,
    c_hidden_dim:int,
    action_dim:int,
    prob:bool
):
    #controller instanse
    controller = Controller(
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        target_dim=target_dim,
        c_hidden_dim=c_hidden_dim,
        action_dim=action_dim,
        bias=True
    )

    #create random tensor
    hidden = torch.randn(batch_size,hidden_dim)
    state = torch.randn(batch_size,state_dim)
    target = torch.randn(batch_size,target_dim)
    c_hidden = torch.randn(batch_size,c_hidden_dim)

    action, next_controller_hidden = controller(hidden,state,target,c_hidden,prob)

    assert next_controller_hidden.size() == torch.Size([batch_size,c_hidden_dim])
    assert action.size() == torch.Size([batch_size,action_dim])