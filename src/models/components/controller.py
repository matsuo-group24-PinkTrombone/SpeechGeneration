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
        hidden_size: int,
        state_size: int,
        encoder_modules: Tuple,
        feats_T: int,
        c_hidden_size: int,
        action_size: int,
        bias: bool = True,
    ):
        """
        Args:
            hidden_size (int): The size of hidden state(h_t) of RNN
            state_size (int): The size of state(s_t)
            encoder_modules (Tuple): posterior_encoder_vitsのencoderモジュールのインスタンス
            c_hidden_size (int): The size of controller_hidden_state(hc_t) of RNN
            action_size (int): The size of action(a_t)
            bias (bool, optional): This argument determines whether RNN requires bias or not. Defaults to True.
        """

        # define target melspectrogram encoder
        # define modules

        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.input_conv, self.encoder, self.encoder_proj = encoder_modules
        self.c_hidden_size = c_hidden_size
        self.action_size = action_size

        self.state_emb = torch.nn.Linear(feats_T, 1)
        self.rnn = torch.nn.GRUCell(
            input_size=hidden_size + state_size * 2,
            hidden_size=c_hidden_size,
            bias=bias,
        )
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(c_hidden_size, action_size), torch.nn.Tanh()
        )

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
            hidden (Tensor):hidden state of RNN[B,hidden_size]
            state (Tensor):state of Observation [B,state_size]
            target (Tensor):target melspectrogram [B,in_channels, T_feats]
            cotroller_hidden (Tensor):hidden state of controller RNN[B,c_hidden_size]
            probabilistic (bool):  If True, sample action from normal distribution.
        """
        # target encoding
        x = self.input_conv(target)
        x = self.encoder(x, None, g=None)
        enc_stats = self.encoder_proj(x)
        enc_m, enc_logs = enc_stats.split(enc_stats.size(1) // 2, dim=1)
        target_state = enc_m + torch.randn_like(enc_m) * torch.exp(enc_logs)

        embed_target_state = self.state_emb(target_state).squeeze(2)

        # RNN
        embed_state = self.state_emb(state).squeeze(2)
        hidden_state = torch.cat((hidden, embed_state), dim=1)
        hidden_state_target = torch.cat((hidden_state, embed_target_state), dim=1)

        next_controller_hidden = self.rnn(hidden_state_target, controller_hidden)

        # action
        if probabilistic:
            action = torch.clip(
                torch.normal(mean=0, std=1, size=(hidden.size(0), self.action_size)), min=-1, max=1
            )
        else:
            assert next_controller_hidden.size() == torch.Size(
                [controller_hidden.size(0), self.c_hidden_size]
            )
            action = self.proj(next_controller_hidden)

        return action, next_controller_hidden

    @property
    def controller_hidden_shape(self):
        return (self.c_hidden_size,)
