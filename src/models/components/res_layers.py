import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias)

    def forward(self, x):
        fx = self.relu(self.fc1(x))
        fx = self.fc2(fx)
        hx = fx + x
        return hx


class ResLayers(nn.Module):
    """ResBlocksの幅と深さと数を変更できるクラス."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        res_hidden_size: int,
        layers: int,
        output_size: int,
        bias: bool = True,
    ) -> None:
        """
        Args:
            input_size (int): The size of input layers.
            hidden_size (int): The size of hidden layers.
            res_hidden_size: The size of hidden layers in ResBlock.
            layers (int): The number of ResBlocks.
            output_size (int): The size of output.
            bias (bool): Whether you need bias or not in ResBlock.
        """
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size, bias=bias)

        self.hidden_layers = nn.Sequential()

        args = (hidden_size, res_hidden_size, hidden_size, bias)
        for _ in range(layers):
            self.hidden_layers.append(ResBlock(*args))
            # self.hidden_layers.append(nn.LayerNorm(hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor

        Returns:
            output: output tensor
        """

        hidden_first = self.input_layer(x)

        hidden_last = self.hidden_layers(hidden_first)

        output = self.output_layer(hidden_last)

        return output
