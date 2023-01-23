import torch
from torch import nn
from src.models.abc.transition import Transition
from src.models.abc._types import _tensor_or_any as _toa

class TransitionModel(Transition):
    def __init__(
        self,
        hidden_size: int,
        state_size: int,
        action_size: int,
        input_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.rnn = nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
        )
        self.fc_action_state = nn.Linear(state_size + action_size, input_size)
        
    def forward(self, hidden:_toa, state:_toa, action:_toa)->torch.Tensor:
        rnn_input = self.fc_action_state(torch.cat((state, action), dim=-1))
        print(rnn_input.shape)
        print(hidden.shape)
        next_hidden = self.rnn(rnn_input, hidden)
        return next_hidden

    @property
    def hidden_shape(self) -> tuple[int]:
        return (self.hidden_size,)

    def __call__(self, hidden:_toa, state:_toa, action:_toa)->torch.Tensor:
        return self.forward(hidden, state, action)