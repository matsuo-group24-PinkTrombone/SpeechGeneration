from typing import Tuple

import torch
from torch import Tensor

from ...utils.nets_utils import make_non_pad_mask
from ..abc.controller import Controller as AbsController
from .wavenet import WaveNet
from .wavenet.residual_block import Conv1d


class Controller(AbsController):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        encoder_modules: Tuple,
        feats_T: int,
        c_hidden_dim: int,
        action_dim: int,
        bias: bool,
    ):
        """
        Args:
            hidden_dim (int):
            state_dim (int):
            c_hidden_dim (int):
        """

        # define target melspectrogram encoder
        # define modules

        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.input_conv, self.encoder, self.encoder_proj = encoder_modules
        self.c_hidden_dim = c_hidden_dim
        self.action_dim = action_dim

        self.state_emb = torch.nn.Linear(feats_T, 1)
        self.rnn = torch.nn.GRUCell(
            input_size=hidden_dim + state_dim * 2,
            hidden_size=c_hidden_dim,
            bias=bias,
        )
        self.proj = torch.nn.Sequential(torch.nn.Linear(c_hidden_dim, action_dim), torch.nn.Tanh())

    def forward(
        self,
        hidden: Tensor,
        state: Tensor,
        target: Tensor,
        controller_hidden: Tensor,
        probabilistic: bool,
    ):
        """
        Args:
            hidden (Tensor):[B,hidden_dim]
            state (Tensor):[B,state_dim]
            target (Tensor):melspectrogram [B,in_channels, T_feats]
            cotroller_hidden (Tensor):[B,c_hidden_dim]
            probabilistic (bool):  If True, sample action from normal distribution.
        """
        # target encoding
        x = self.input_conv(target)
        x = self.encoder(x, None, g=None)
        enc_stats = self.encoder_proj(x)
        enc_m, enc_logs = enc_stats.split(enc_stats.size(1) // 2, dim=1)
        target_state = enc_m + torch.randn_like(enc_m) * torch.exp(enc_logs)

        t_state_emb = self.state_emb(target_state).squeeze(2)

        # RNN
        state_emb = self.state_emb(state).squeeze(2)
        hidden_state = torch.cat((hidden, state_emb), dim=1)
        hidden_state_target = torch.cat((hidden_state, t_state_emb), dim=1)

        next_controller_hidden = self.rnn(hidden_state_target, controller_hidden)

        # action
        if probabilistic:
            action = torch.clip(
                torch.normal(mean=0, std=1, size=(hidden.size(0), self.action_dim)), min=-1, max=1
            )
        else:
            assert next_controller_hidden.size() == torch.Size(
                [controller_hidden.size(0), self.c_hidden_dim]
            )
            action = self.proj(next_controller_hidden)

        return action, next_controller_hidden

    @property
    def controller_hidden_shape(self):
        return self.c_hidden_dim
