import torch
from torch import Tensor, nn

from ...abc.transition import Transition as AbstractTransition
from ..linear_layers import LinearLayers


class Transition(AbstractTransition):
    def __init__(
        self,
        hidden_size: int,
        state_size: int,
        action_size: int,
        input_size: int,
        stacked_linear_hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
    ):
        """
        Args:
            hidden_size (int): The size of hidden state(h_t) of RNN
            state_size (int): The size of state(s_t)
            action_size (int): The size of action(a_t)
            input_size (int): The size of vector input to RNN
            stacked_linear_hidden_size (int): The hidden size of `fc_input_layer`.
            num_layers (int): The number of internal layers (For stacked linear).
            bias (bool, optional): This argument determines whether RNN requires bias or not. Defaults to True.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = input_size
        self.stacked_linear_hidden_size = stacked_linear_hidden_size
        self.bias = bias

        self.fc_action_state = LinearLayers(
            self.state_size + self.action_size,
            stacked_linear_hidden_size,
            num_layers,
            input_size,
            bias,
        )

        self.rnn = nn.GRUCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bias=bias,
        )

    def forward(self, hidden: Tensor, state: Tensor, action: Tensor) -> Tensor:
        """Method for computing f(h_t, s_t, a_t) -> h_t+1.

        Args:
            hidden (Tensor): hidden state of RNN(h_t)
            state (Tensor): state of Observation (s_t)
            action (Tensor): action (a_t)

        Returns:
            Tensor: next hidden state of RNN(h_t+1)
        """
        rnn_input = self.fc_action_state(torch.cat((state, action), dim=-1))
        next_hidden = self.rnn(rnn_input, hidden)
        return next_hidden

    @property
    def hidden_shape(self) -> tuple[int]:
        return (self.hidden_size,)
