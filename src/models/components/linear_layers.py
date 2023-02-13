import torch
import torch.nn as nn


class LinearLayers(nn.Module):
    """Linearの幅と深さを変更できるクラス."""

    def __init__(
        self, input_size: int, hidden_size: int, layers: int, output_size: int, bias: bool = True
    ) -> None:
        """
        Args:
            input_size (int):
            hidden_size (int):
            layers (int):
            output_size (int):
            bias (bool):
        """
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size, bias=bias)

        self.hidden_layers = nn.Sequential()

        for _ in range(layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
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
