import torch
from torch import nn

from src.models.abc._types import _tensor_or_any as _toa
from src.models.abc.transition import Transition as AbstractTransition


class Transition(AbstractTransition):
    def __init__(
        self,
        hidden_size: int,
        state_size: int,
        action_size: int,
        input_size: int,
        bias: bool = True,
    ):
        """
        Args:
            hidden_size (int): The size of hidden state(h_t) of RNN
            state_size (int): The size of state(s_t)
            action_size (int): The size of action(a_t)
            input_size (int): The size of vector input to RNN
            bias (bool, optional): This argument determines whether RNN requires bias or not. Defaults to True.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = input_size
        self.bias = bias
        self.rnn = nn.GRUCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bias=bias,
        )
        self.fc_action_state = nn.Linear(self.state_size + self.action_size, self.input_size)

    def forward(self, hidden: _toa, state: _toa, action: _toa) -> torch.Tensor:
        """Method for computing f(h_t, s_t, a_t) -> h_t+1.

        Args:
            hidden (_toa): hidden state of RNN(h_t)
            state (_toa): state of Observation (s_t)
            action (_toa): action (a_t)

        Returns:
            torch.Tensor: next hidden state of RNN(h_t+1)
        """
        rnn_input = self.fc_action_state(torch.cat((state, action), dim=-1))
        print(rnn_input.shape)
        print(hidden.shape)
        next_hidden = self.rnn(rnn_input, hidden)
        return next_hidden

    @property
    def hidden_shape(self) -> tuple[int]:
        return (self.hidden_size,)

    def __call__(self, hidden: _toa, state: _toa, action: _toa) -> torch.Tensor:
        return self.forward(hidden, state, action)
