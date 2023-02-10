from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ...abc.controller import Controller as AbsController
from ..linear_layers import LinearLayers
from ..posterior_encoder_vits import PosteriorEncoderVITS


class Controller(AbsController):
    def __init__(
        self,
        encoder: PosteriorEncoderVITS,
        hidden_size: int,
        state_size: int,
        c_hidden_size: int,
        action_size: int,
        input_size: int,
        feats_T: int,
        stacked_linear_hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
    ) -> None:
        """
        Args:
            encoder (PosteriorEncoderVITS): target encoder module.
            hidden_size (int): The size of hidden state(h_t) of RNN
            state_size (int): The size of state(s_t)
            c_hidden_size (int): The size of controller_hidden_state(hc_t) of RNN
            action_size (int): The size of action(a_t)
            input_size (int): Input dimension of RNN
            stacked_linear_hidden_size (int): The hidden size of `fc_input_layer`.
            num_layers (int): The number of internal layers (For stacked linear).
            bias (bool, optional): This argument determines whether RNN requires bias or not. Defaults to True.
        """

        # define target melspectrogram encoder
        # define modules

        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.encoder = encoder
        self.feats_T = feats_T
        self.c_hidden_size = c_hidden_size
        self.stacked_linear_hidden_size = stacked_linear_hidden_size
        self.num_layers = num_layers
        self.action_size = action_size

        self.fc_time_reduction = nn.Linear(feats_T, 1)

        self.fc_input_layer = LinearLayers(
            hidden_size + state_size + state_size,  # hidden + obs_state + target_state
            stacked_linear_hidden_size,
            num_layers,
            input_size,
            bias,
        )

        self.rnn = nn.GRUCell(
            input_size=input_size,
            hidden_size=c_hidden_size,
            bias=bias,
        )

        self.fc_mean = nn.Linear(hidden_size, action_size, bias)
        self.fc_logs = nn.Linear(hidden_size, action_size, bias)

    def forward(
        self,
        hidden: Tensor,
        state: Tensor,
        target: Tensor,
        controller_hidden: Tensor,
        probabilistic: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden (Tensor):hidden state of RNN[B,hidden_size]
            state (Tensor):state of Observation [B,state_size]
            target (Tensor):target melspectrogram [B,in_channels, T_feats]
            cotroller_hidden (Tensor):hidden state of controller RNN[B,c_hidden_size]
            probabilistic (bool):  If True, sample action from normal distribution.

        Returns:
            action (Tensor): ction data `a_t`. The value range must be [-1, 1].
            next_controller_hidden (Tensor): Next controller hidden state `hc_{t+1}`.
        """
        # target encoding
        _, _m, _, _ = self.encoder(
            target, torch.tensor([self.feats_T] * hidden.size(0), dtype=torch.float32)
        )
        target_state = self.fc_time_reduction(_m).squeeze(2)

        # RNN
        hidden_state = torch.cat((hidden, state), dim=1)
        hidden_state_target = torch.cat((hidden_state, target_state), dim=1)

        rnn_input = self.fc_input_layer(hidden_state_target)
        next_controller_hidden = self.rnn(rnn_input, controller_hidden)

        # action
        m = self.fc_mean(next_controller_hidden)
        logs = self.fc_logs(next_controller_hidden)

        if probabilistic:
            action = m + torch.randn_like(m) * torch.exp(logs)
        else:
            action = m

        return torch.tanh(action), next_controller_hidden

    @property
    def controller_hidden_shape(self):
        return (self.c_hidden_size,)
