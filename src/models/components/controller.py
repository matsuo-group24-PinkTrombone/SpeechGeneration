import torch
from torch import Tensor
from ..abc.controller import Controller as AbsController


class Controller(AbsController):
    def __init__(
        self,
        hidden_dim:int,
        state_dim:int,
        target_dim:int,
        c_hidden_dim:int,
        action_dim:int,
        bias:bool
    ):
        """
        Args:
            hidden_dim (int):
            state_dim (int):
            target_dim (int):
            c_hidden_dim (int):
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.target_dim = target_dim
        self.c_hidden_dim = c_hidden_dim

        self.rnn = torch.nn.GRUCell(
            input_size=hidden_dim+state_dim+target_dim,
            hidden_size=c_hidden_dim,
            bias=bias,
        )
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(
                c_hidden_dim,
                action_dim*2
            ),
            torch.nn.Tanh()
        )
        
    
    def forward(
        self,
        hidden: Tensor,
        state: Tensor,
        target: Tensor,
        controller_hidden: Tensor,
        probablistic: bool
    ):
        """
        Args:
            hidden (Tensor):[B,hidden_dim]
            state (Tensor):[B,state_dim]
            target (Tensor):[B,target_dim]
            cotroller_hidden (Tensor):[B,c_hidden_dim]
            probablistic (bool):  If True, sample action from normal distribution.
        """
        # RNN
        hidden_state = torch.cat((hidden, state),dim=1)
        hidden_state_target = torch.cat((hidden_state,target),dim=1)

        next_controller_hidden = self.rnn(hidden_state_target,controller_hidden)
        
        # action
        stats = self.proj(next_controller_hidden)
        m, logs = stats.split(stats.size(1) // 2, dim=1)
        if probablistic:
            action = (m + torch.randn_like(m) * torch.exp(logs))
        else:
            action = m

        return action, next_controller_hidden
    
    @property
    def controller_hidden_shape(self):
        return self.c_hidden_dim